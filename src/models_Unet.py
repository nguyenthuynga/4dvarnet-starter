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



class BilinAE_Unet(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True):
        super().__init__()
        self.bilin_quad = bilin_quad
        self.conv_in = nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.bilin_1 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_21 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_22 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.conv_out = nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def forward(self, x):
        # print(f"Shape input: {x.shape}")
        x = self.down(x)
        # print(f"After Down : {x.shape}")
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        # print(f"After nonlin : {nonlin.shape}")
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        # print(f"After cat : {x.shape}")
        x = self.up(x)
        # print(f"After up : {x.shape}")
        return x




rateDropout = 0.2
padding_mode = 'reflect'

class DoubleConvBILIN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,padding_mode='reflect'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.conv1  = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
        self.conv21 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
        self.conv22 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
        self.conv23 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
        self.conv24 = nn.Conv2d(2*mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)

        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu( self.bn1(x1) )
        
        x11 = self.conv21(x1) 
        x12 = self.conv22(x1) 
        x13 = self.conv23(x1) 
        x1 = self.conv24( torch.cat((x11,x12*x13),dim=1) )
        
        x1 = self.bn2(x1)
        
        return x1
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,padding_mode='reflect',activation='relu'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        if activation == 'relu':
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(rateDropout),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
        elif activation == 'tanh' :
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(mid_channels),
                    nn.Tanh(inplace=True),
                    nn.Dropout(rateDropout),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(out_channels),
                    nn.Tanh(inplace=True) )
        elif activation == 'logsigmoid' :
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(mid_channels),
                    nn.LogSigmoid(inplace=True),
                    nn.Dropout(rateDropout),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(out_channels),
                    nn.LogSigmoid(inplace=True) )
        elif activation == 'bilin' :
            self.double_conv = DoubleConvBILIN(in_channels, mid_channels,padding_mode=padding_mode)

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module): #a standard UNet from Ronan
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

