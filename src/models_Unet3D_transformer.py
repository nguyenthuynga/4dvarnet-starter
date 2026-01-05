import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F


"""changes: 
1, change path_trains in base config to address the dataset for OSSE
2, log the metrics so that don't have to redo it as post processing in def step function
3, create compute_rmse and compute_re functions and then change step to include these two metric computation, add epsilon in compute_re to avoid dividing by zero
"""


#out = self.forward(batch.input)# change here since the original here not work: self(batch=batch)
#I got NaN on metrics but not on loss, so I rewirte metrics RMSE and RE here,but I think the results of those two metrics are not different compared to the old


#same as model I think, I just change the way to write rmse and re, but I think the results of those two metrics are not different
def compute_rmse(predicted, target, input):
    # Mask where target is not NaN and input is NaN
    valid_mask = (~torch.isnan(target).bool()) & torch.isnan(input).bool()
    valid_predictions = predicted[valid_mask]
    valid_targets = target[valid_mask]
    mse = torch.nanmean((valid_targets - valid_predictions) ** 2)
    return torch.sqrt(mse)

def compute_re(predicted, target, input):
    # Mask where target is not NaN and input is NaN
    valid_mask = (~torch.isnan(target).bool()) & torch.isnan(input).bool()
    valid_predictions = predicted[valid_mask]
    valid_targets = target[valid_mask]
    epsilon = 1e-8  # Small constant to avoid division by zero
    re = torch.nanmean(torch.abs(10**valid_targets - 10**valid_predictions) / (10**valid_targets + epsilon)) * 100
    return re

class Lit4dVarNet(pl.LightningModule):
    def __init__(self, unet, rec_weight, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True):
        super().__init__()
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.test_data = None
        self.unet=unet
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)

    def print_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")

    @property
    def norm_stats(self):#ko biet de lam gi
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0., 1.)

    @staticmethod
    def weighted_mse(err, weight):#computes the MSE loss only for the valid, weighted error elements.
        err_w = err * weight[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    def training_step(self, batch, batch_idx):
        #changes: print things
        # print("Training Step - Batch type:", type(batch))
        # print("Training Step - Batch content:", batch)
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]

    def forward(self, x):
        return self.unet(x)
    
    def step(self, batch, phase=""):
        #if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            #return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)

        self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Compute custom metrics with masking
        input=batch.input
        target=batch.tgt
        rmse = compute_rmse(out, target, input)
        re = compute_re(out, target, input)

        # Log metrics
        self.log(f'{phase}_rmse', rmse, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{phase}_re', re, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss 
        # print(f"50 * loss {50 * loss} + 1000 * grad_loss {1000 * grad_loss}")
        return training_loss, out
    

    def base_step(self, batch, phase=""):
        out = self.forward(batch.input.nan_to_num())# change here since the original here not work: self(batch=batch), and remember batch.input.nan_to_num() to avoid NaN
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        # out = self(batch=batch)
        out = self.forward(batch.input.nan_to_num())# change here since the original here not work: self(batch=batch), and remember batch.input.nan_to_num() to avoid NaN
        m, s = self.norm_stats

        self.test_data.append(torch.stack(
            [
                batch.input.cpu() * s + m,
                batch.tgt.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))

    @property
    def test_quantities(self):
        return ['inp', 'tgt', 'out']

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight.cpu().numpy()
        )

        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        self.test_data = rec_da.assign_coords(
            dict(v0=self.test_quantities)
        ).to_dataset(dim='v0')

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data) 
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            print(Path(self.trainer.log_dir) / 'test_data.nc')
            self.logger.log_metrics(metrics.to_dict())



# class BilinAE_Unet(nn.Module):
#     def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True):
#         super().__init__()
#         self.bilin_quad = bilin_quad
#         self.conv_in = nn.Conv2d(
#             dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
#         )
#         self.conv_hidden = nn.Conv2d(
#             dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
#         )

#         self.bilin_1 = nn.Conv2d(
#             dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
#         )
#         self.bilin_21 = nn.Conv2d(
#             dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
#         )
#         self.bilin_22 = nn.Conv2d(
#             dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
#         )

#         self.conv_out = nn.Conv2d(
#             2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
#         )

#         self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
#         self.up = (
#             nn.UpsamplingBilinear2d(scale_factor=downsamp)
#             if downsamp is not None
#             else nn.Identity()
#         )

#     def forward(self, x):
#         # print(f"Shape input: {x.shape}")
#         x = self.down(x)
#         # print(f"After Down : {x.shape}")
#         x = self.conv_in(x)
#         x = self.conv_hidden(F.relu(x))

#         nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
#         # print(f"After nonlin : {nonlin.shape}")
#         x = self.conv_out(
#             torch.cat([self.bilin_1(x), nonlin], dim=1)
#         )
#         # print(f"After cat : {x.shape}")
#         x = self.up(x)
#         # print(f"After up : {x.shape}")
#         return x




# rateDropout = 0.2
# padding_mode = 'reflect'

# class DoubleConvBILIN(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None,padding_mode='reflect'):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
            
#         self.conv1  = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
#         self.conv21 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
#         self.conv22 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
#         self.conv23 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
#         self.conv24 = nn.Conv2d(2*mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)

#         self.bn1 = nn.BatchNorm2d(mid_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x1 = F.relu( self.bn1(x1) )
        
#         x11 = self.conv21(x1) 
#         x12 = self.conv22(x1) 
#         x13 = self.conv23(x1) 
#         x1 = self.conv24( torch.cat((x11,x12*x13),dim=1) )
        
#         x1 = self.bn2(x1)
        
#         return x1
    
# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None,padding_mode='reflect',activation='relu'):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
            
#         if activation == 'relu':
#             self.double_conv = nn.Sequential(
#                     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
#                     nn.BatchNorm2d(mid_channels),
#                     nn.ReLU(inplace=True),
#                     nn.Dropout(rateDropout),
#                     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(inplace=True)
#                 )
#         elif activation == 'tanh' :
#             self.double_conv = nn.Sequential(
#                     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
#                     nn.BatchNorm2d(mid_channels),
#                     nn.Tanh(inplace=True),
#                     nn.Dropout(rateDropout),
#                     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
#                     nn.BatchNorm2d(out_channels),
#                     nn.Tanh(inplace=True) )
#         elif activation == 'logsigmoid' :
#             self.double_conv = nn.Sequential(
#                     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
#                     nn.BatchNorm2d(mid_channels),
#                     nn.LogSigmoid(inplace=True),
#                     nn.Dropout(rateDropout),
#                     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
#                     nn.BatchNorm2d(out_channels),
#                     nn.LogSigmoid(inplace=True) )
#         elif activation == 'bilin' :
#             self.double_conv = DoubleConvBILIN(in_channels, mid_channels,padding_mode=padding_mode)

#     def forward(self, x):
#         return self.double_conv(x)
    
# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             #nn.AvgPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)
    

# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)
  
  
  
  
  
  
  
  
  
  
  
  
  
############### The UNet3D Temporal Attention ###############  
# unet_film_temporalattn_3d.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utilities ----------

def _choose_gn_groups(c: int, prefer: int = 32) -> int:
    """Pick the largest number of groups <= prefer that divides c."""
    for g in [prefer, 16, 8, 4, 2, 1]:
        if c % g == 0:
            return g
    return 1

# ---------- Core building blocks ----------

class ResBlock3D_FiLM(nn.Module):
    """
    3D residual block with FiLM time conditioning and dropout.
    - Two 3x3x3 convs, GroupNorm, SiLU.
    - FiLM provides per-channel (scale, shift) after first norm.
    - Optional channel change via 1x1x1 skip.
    - Zero-init last conv for stable residual learning.
    """
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.10):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.norm1 = nn.GroupNorm(_choose_gn_groups(c_in), c_in)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=3, padding=1)

        # self.film = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * c_out))

        self.norm2 = nn.GroupNorm(_choose_gn_groups(c_out), c_out)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=3, padding=1)

        self.skip = nn.Identity() if c_in == c_out else nn.Conv3d(c_in, c_out, kernel_size=1)

        # Zero-init final conv to encourage identity at start
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T, H, W)
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)

        return h + self.skip(x)

class TemporalSelfAttention3D(nn.Module):
    """
    Self-attention along the temporal axis only, with pre-norm and residual.
    - Flattens spatial dims (H*W) into batch for efficiency.
    """
    def __init__(self, channels: int, n_heads: int = 4, attn_dropout: float = 0.20):
        super().__init__()
        self.norm = nn.GroupNorm(_choose_gn_groups(channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        returns: (B, C, T, H, W)
        """
        b, c, t, h, w = x.shape
        h_in = x
        x = self.norm(x)

        # (B, C, T, H, W) -> (B*H*W, T, C)
        x = x.permute(0, 3, 4, 2, 1).contiguous().view(b * h * w, t, c)
        x_out, _ = self.attn(x, x, x, need_weights=False)
        x_out = self.drop(x_out)
        # back to (B, C, T, H, W)
        x_out = x_out.view(b, h, w, t, c).permute(0, 4, 3, 1, 2).contiguous()
        return h_in + x_out

class Downsample3D(nn.Module):
    """
    Simple strided 3D conv downsample that also sets the target channel count.
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample3D(nn.Module):
    """
    3D transposed-conv upsample to target channel count.
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.tconv = nn.ConvTranspose3d(
            c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tconv(x)

# ---------- The UNet ----------

############### ################################################################################################
# 3D U-Net with:
# Flexible input size, no need to be divisble by 8 because of downsampling with stride=2. 
# This lets you use any T,H,W. Pad to multiples of 8 before the network, then unpad the output.
################################################################################################################
def _pad_to_multiple(x, mult=(8,8,8)):
    # x: (B,C,T,H,W); mult: (mT,mH,mW)
    _,_,T,H,W = x.shape
    mT,mH,mW = mult
    pT = (mT - T % mT) % mT
    pH = (mH - H % mH) % mH
    pW = (mW - W % mW) % mW
    # pad order: (W_left, W_right, H_left, H_right, T_left, T_right)
    x = F.pad(x, (0,pW, 0,pH, 0,pT))
    return x, (pT,pH,pW)

def _unpad(x, pads):
    pT,pH,pW = pads
    if pT or pH or pW:
        T,H,W = x.shape[-3:]
        x = x[..., :T-pT, :H-pH, :W-pW]
    return x

class UNet3D_transformer_FlowMatching(nn.Module):
    """
    3D U-Net with:
      - Residual blocks + FiLM time conditioning
      - Two ResBlocks per stage
      - Temporal self-attention at the bottleneck
      - Depth over width: channels = [48, 96, 192, 384]
    Flow-matching ready: no score-based output normalization.

    Inputs:
      x: (B, 1, T, H, W)
      y: (B, 1, T, H, W)  (conditioning; concatenated with x along channel dim)
      t: (B,) or (B,1)    (continuous "time" / flow parameter in [0,1])

    Output:
      (B, 1, T, H, W)
    """
    def __init__(
        self,
        marginal_prob_std=None,             # kept for compatibility; unused in FM
        channels=(48, 96, 192, 384),        # depth over width
        dropout=0.10,                       # set to 0.05–0.20 per your dataset size
        attn_heads=4,
        text_dim=1,                         # kept for signature compatibility
        nAttr=40,
        n_in_channels=10# kept for signature compatibility
    ):
        super().__init__()
        self.channels = list(channels)
        self.dropout = dropout
        self.attn_heads = attn_heads
        self.marginal_prob_std = marginal_prob_std  # not used (FM)


        c1, c2, c3, c4 = self.channels
        self.n_channels = n_in_channels

        # --- Encoder (Down path): 2 ResBlocks per stage ---
        # Stage 1
        self.enc1_block1 = ResBlock3D_FiLM(c_in=self.n_channels,  c_out=c1,  dropout=dropout)
        self.enc1_block2 = ResBlock3D_FiLM(c_in=c1, c_out=c1,  dropout=dropout)
        self.down1 = Downsample3D(c_in=c1, c_out=c2)

        # Stage 2
        self.enc2_block1 = ResBlock3D_FiLM(c_in=c2, c_out=c2, dropout=dropout)
        self.enc2_block2 = ResBlock3D_FiLM(c_in=c2, c_out=c2, dropout=dropout)
        self.down2 = Downsample3D(c_in=c2, c_out=c3)

        # Stage 3
        self.enc3_block1 = ResBlock3D_FiLM(c_in=c3, c_out=c3, dropout=dropout)
        self.enc3_block2 = ResBlock3D_FiLM(c_in=c3, c_out=c3,dropout=dropout)
        self.down3 = Downsample3D(c_in=c3, c_out=c4)

        # --- Bottleneck ---
        self.bot_block1 = ResBlock3D_FiLM(c_in=c4, c_out=c4, dropout=dropout)
        self.bot_attn   = TemporalSelfAttention3D(channels=c4, n_heads=attn_heads, attn_dropout=dropout)
        self.bot_block2 = ResBlock3D_FiLM(c_in=c4, c_out=c4, dropout=dropout)

        # --- Decoder (Up path): upsample -> concat skip -> 2 ResBlocks ---
        # Up from bottleneck to stage 3
        self.up3 = Upsample3D(c_in=c4, c_out=c3)
        self.dec3_block1 = ResBlock3D_FiLM(c_in=c3 + c3, c_out=c3,  dropout=dropout)
        self.dec3_block2 = ResBlock3D_FiLM(c_in=c3,       c_out=c3,dropout=dropout)

        # Up to stage 2
        self.up2 = Upsample3D(c_in=c3, c_out=c2)
        self.dec2_block1 = ResBlock3D_FiLM(c_in=c2 + c2, c_out=c2, dropout=dropout)
        self.dec2_block2 = ResBlock3D_FiLM(c_in=c2,       c_out=c2, dropout=dropout)

        # Up to stage 1
        self.up1 = Upsample3D(c_in=c2, c_out=c1)
        self.dec1_block1 = ResBlock3D_FiLM(c_in=c1 + c1, c_out=c1,dropout=dropout)
        self.dec1_block2 = ResBlock3D_FiLM(c_in=c1,       c_out=c1, dropout=dropout)

        # --- Final head ---
        self.out_norm = nn.GroupNorm(_choose_gn_groups(c1), c1)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv3d(c1, 1, kernel_size=3, padding=1)

        # (Optional) attribute embedding kept for signature parity; not used here
        self.cond_embed = nn.Embedding(nAttr + 1, text_dim, padding_idx=nAttr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate observed/conditioning y with target x along channel dim
        # if y is None:
        #     # If y is missing, use zeros of same shape as x
        #     y = torch.zeros_like(x)
        # h = torch.cat([x, y], dim=1)  # (B, 2, T, H, W)
        # print(f"Input shape: {x.shape}")
        
        x = x.unsqueeze(1)
        # print(f"Shape after unsqueeze: {x.shape}")
        
        (x, pads) = _pad_to_multiple(x, (8,8,8))
        # print(f"Input shape after padding: {x.shape}")
        
        h = x  # (B, 1, T, H, W)

        # ----- Encoder -----
        h1 = self.enc1_block1(h)
        # print(f"Shape after enc1_block1: {h1.shape}")
        h1 = self.enc1_block2(h1)
        d1 = self.down1(h1)
        # print(f"Shape after down1: {d1.shape}")

        h2 = self.enc2_block1(d1)
        # print(f"Shape after enc2_block1: {h2.shape}")
        h2 = self.enc2_block2(h2)
        # print(f"Shape after enc2_block2: {h2.shape}")
        d2 = self.down2(h2)
        # print(f"Shape after down2: {d2.shape}")

        h3 = self.enc3_block1(d2)
        # print(f"Shape after enc3_block1: {h3.shape}")
        h3 = self.enc3_block2(h3)
        d3 = self.down3(h3)
        # print(f"Shape after down3: {d3.shape}")

        # ----- Bottleneck -----
        b = self.bot_block1(d3)
        # print(f"Shape after bot_block1: {b.shape}")
        b = self.bot_attn(b)
        # print(f"Shape after bot_attn: {b.shape}")
        b = self.bot_block2(b)
        # print(f"Shape after bot_block2: {b.shape}")

        # ----- Decoder -----
        u3 = self.up3(b)
        # print(f"Shape after up3: {u3.shape}")
        u3 = torch.cat([u3, h3], dim=1)
        # print(f"Shape after concat with h3: {u3.shape}")
        u3 = self.dec3_block1(u3)
        # print(f"Shape after dec3_block1: {u3.shape}")
        u3 = self.dec3_block2(u3)
        # print(f"Shape after dec3_block2: {u3.shape}")

        u2 = self.up2(u3)
        # print(f"Shape after up2: {u2.shape}")
        u2 = torch.cat([u2, h2], dim=1)
        # print(f"Shape after concat with h2: {u2.shape}")
        u2 = self.dec2_block1(u2)
        # print(f"Shape after dec2_block1: {u2.shape}")
        u2 = self.dec2_block2(u2)
        # print(f"Shape after dec2_block2: {u2.shape}")

        u1 = self.up1(u2)
        # print(f"Shape after up1: {u1.shape}")
        u1 = torch.cat([u1, h1], dim=1)
        # print(f"Shape after concat with h1: {u1.shape}")
        u1 = self.dec1_block1(u1)
        # print(f"Shape after dec1_block1: {u1.shape}")
        u1 = self.dec1_block2(u1)
        # print(f"Shape after dec1_block2: {u1.shape}")

        out = self.out_norm(u1)
        # print(f"Shape after out_norm: {out.shape}")
        out = self.out_act(out)
        # print(f"Shape after out_act: {out.shape}")
        out = self.out_conv(out)
        # print(f"Output shape: {out.shape}")
        
        out = _unpad(out, pads)
        # print(f"Output shape after unpad: {out.shape}")

        out = out.squeeze(1)  # Remove the channel dimension after UNet
        # print(f"Output shape after squeeze: {out.shape}")

        # Flow matching: no division by marginal_prob_std(t)
        return out
    

# ---------- The smaller UNet Paul used (before removing additional input y, 
# before add customize input size, and with channels: [64,128,128,256] instead of increasing doublely, 
# and replace ResBlock3D_FiLM_Paul by ResBlock3D_FiLM
# and replace batch by x----------

class ResBlock3D_FiLM_Paul(nn.Module):
    """
    3D residual block with FiLM time conditioning and dropout.
    - Two 3x3x3 convs, GroupNorm, SiLU.
    - FiLM provides per-channel (scale, shift) after first norm.
    - Optional channel change via 1x1x1 skip.
    - Zero-init last conv for stable residual learning.
    """
    def __init__(self, c_in: int, c_out: int, t_dim: int, dropout: float = 0.10):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.norm1 = nn.GroupNorm(_choose_gn_groups(c_in), c_in)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=3, padding=1)

        self.film = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * c_out))

        self.norm2 = nn.GroupNorm(_choose_gn_groups(c_out), c_out)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=3, padding=1)

        self.skip = nn.Identity() if c_in == c_out else nn.Conv3d(c_in, c_out, kernel_size=1)

        # Zero-init final conv to encourage identity at start
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T, H, W)
        t_emb: (B, t_dim)
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # FiLM conditioning
        h = self.norm2(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)

        return h + self.skip(x)
    
class UNet_Tranformer_attrb1_smallermodel_byPaul(nn.Module):
    """
    3D U-Net with:
      - Residual blocks + FiLM time conditioning
      - Two ResBlocks per stage
      - Temporal self-attention at the bottleneck
      - Depth over width: channels = [48, 96, 192, 384]
    Flow-matching ready: no score-based output normalization.

    Inputs:
      x: (B, 1, T, H, W)
      y: (B, 1, T, H, W)  (conditioning; concatenated with x along channel dim)
      t: (B,) or (B,1)    (continuous "time" / flow parameter in [0,1])

    Output:
      (B, 1, T, H, W)
    """
    def __init__(
        self,
        marginal_prob_std=None,             # kept for compatibility; unused in FM
        channels=[64,128,128,256],        # depth over width # channels=(48, 96, 192, 384)
        embed_dim=256,
        dropout=0.10,                       # set to 0.05–0.20 per your dataset size
        attn_heads=4,
        text_dim=1,                         # kept for signature compatibility
        nAttr=40                            # kept for signature compatibility
    ):
        super().__init__()
        self.channels = list(channels)
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.attn_heads = attn_heads
        self.marginal_prob_std = marginal_prob_std  # not used (FM)

        # --- Time embedding ---

        c1, c2, c3, c4 = self.channels

        # --- Encoder (Down path): 2 ResBlocks per stage ---
        # Stage 1
        self.enc1_block1 = ResBlock3D_FiLM_Paul(c_in=2,  c_out=c1, t_dim=embed_dim, dropout=dropout)
        self.enc1_block2 = ResBlock3D_FiLM_Paul(c_in=c1, c_out=c1, t_dim=embed_dim, dropout=dropout)
        self.down1 = Downsample3D(c_in=c1, c_out=c2)

        # Stage 2
        self.enc2_block1 = ResBlock3D_FiLM_Paul(c_in=c2, c_out=c2, t_dim=embed_dim, dropout=dropout)
        self.enc2_block2 = ResBlock3D_FiLM_Paul(c_in=c2, c_out=c2, t_dim=embed_dim, dropout=dropout)
        self.down2 = Downsample3D(c_in=c2, c_out=c3)

        # Stage 3
        self.enc3_block1 = ResBlock3D_FiLM_Paul(c_in=c3, c_out=c3, t_dim=embed_dim, dropout=dropout)
        self.enc3_block2 = ResBlock3D_FiLM_Paul(c_in=c3, c_out=c3, t_dim=embed_dim, dropout=dropout)
        self.down3 = Downsample3D(c_in=c3, c_out=c4)

        # --- Bottleneck ---
        self.bot_block1 = ResBlock3D_FiLM_Paul(c_in=c4, c_out=c4, t_dim=embed_dim, dropout=dropout)
        self.bot_attn   = TemporalSelfAttention3D(channels=c4, n_heads=attn_heads, attn_dropout=dropout)
        self.bot_block2 = ResBlock3D_FiLM_Paul(c_in=c4, c_out=c4, t_dim=embed_dim, dropout=dropout)

        # --- Decoder (Up path): upsample -> concat skip -> 2 ResBlocks ---
        # Up from bottleneck to stage 3
        self.up3 = Upsample3D(c_in=c4, c_out=c3)
        self.dec3_block1 = ResBlock3D_FiLM_Paul(c_in=c3 + c3, c_out=c3, t_dim=embed_dim, dropout=dropout)
        self.dec3_block2 = ResBlock3D_FiLM_Paul(c_in=c3,       c_out=c3, t_dim=embed_dim, dropout=dropout)

        # Up to stage 2
        self.up2 = Upsample3D(c_in=c3, c_out=c2)
        self.dec2_block1 = ResBlock3D_FiLM_Paul(c_in=c2 + c2, c_out=c2, t_dim=embed_dim, dropout=dropout)
        self.dec2_block2 = ResBlock3D_FiLM_Paul(c_in=c2,       c_out=c2, t_dim=embed_dim, dropout=dropout)

        # Up to stage 1
        self.up1 = Upsample3D(c_in=c2, c_out=c1)
        self.dec1_block1 = ResBlock3D_FiLM_Paul(c_in=c1 + c1, c_out=c1, t_dim=embed_dim, dropout=dropout)
        self.dec1_block2 = ResBlock3D_FiLM_Paul(c_in=c1,       c_out=c1, t_dim=embed_dim, dropout=dropout)

        # --- Final head ---
        self.out_norm = nn.GroupNorm(_choose_gn_groups(c1), c1)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv3d(c1, 1, kernel_size=3, padding=1)

        # (Optional) attribute embedding kept for signature parity; not used here
        self.cond_embed = nn.Embedding(nAttr + 1, text_dim, padding_idx=nAttr)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        # Concatenate observed/conditioning y with target x along channel dim

        # x = batch.input
        
        x = x.nan_to_num()
        x = x.unsqueeze(1)
        
        (x, pads) = _pad_to_multiple(x, (8,8,8))
        
        y = torch.zeros_like(x)
        h = torch.cat([x, y], dim=1)  # (B, 2, T, H, W)


        # ----- Encoder -----
        h1 = self.enc1_block1(h)
        h1 = self.enc1_block2(h1)
        d1 = self.down1(h1)

        h2 = self.enc2_block1(d1)
        h2 = self.enc2_block2(h2)
        d2 = self.down2(h2)

        h3 = self.enc3_block1(d2)
        h3 = self.enc3_block2(h3)
        d3 = self.down3(h3)

        # ----- Bottleneck -----
        b = self.bot_block1(d3)
        b = self.bot_attn(b)
        b = self.bot_block2(b)

        # ----- Decoder -----
        u3 = self.up3(b)
        # print('h3 :',h3.shape)
        # print('u3 :',u3.shape)
        u3 = torch.cat([u3, h3], dim=1)
        u3 = self.dec3_block1(u3)
        u3 = self.dec3_block2(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.dec2_block1(u2)
        u2 = self.dec2_block2(u2)

        u1 = self.up1(u2)
        # print('h1 :',h1.shape)
        # print('u1 :',u1.shape)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.dec1_block1(u1)
        u1 = self.dec1_block2(u1)

        out = self.out_norm(u1)
        out = self.out_act(out)
        out = self.out_conv(out)

        out = _unpad(out, pads)
        out = torch.squeeze(out, dim=1)

        # Flow matching: no division by marginal_prob_std(t)
        return out

