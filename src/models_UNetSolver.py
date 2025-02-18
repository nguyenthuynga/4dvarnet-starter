import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rmse(predicted, target, input):
    valid_mask = (1 - torch.isnan(target).float()) * torch.isnan(input).float()
    valid_predictions = predicted[valid_mask.bool()]
    valid_targets = target[valid_mask.bool()]
    mse = torch.nanmean((valid_targets - valid_predictions) ** 2)
    return torch.sqrt(mse)

def compute_re(predicted, target, input):
    valid_mask = (1 - torch.isnan(target).float()) * torch.isnan(input).float()
    valid_predictions = predicted[valid_mask.bool()]
    valid_targets = target[valid_mask.bool()]
    epsilon = 1e-8  # Small constant to avoid division by zero
    re = torch.nanmean(torch.abs(10**valid_targets - 10**valid_predictions) / (10**valid_targets + epsilon)) * 100
    return re


class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, rec_weight, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True):
        super().__init__()
        self.solver = solver
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.test_data = None
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

    def forward(self, batch):
        return self.solver(batch)
    
    def step(self, batch, phase=""):
        #if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            #return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Compute custom metrics with masking
        input=batch.input
        target=batch.tgt
        rmse = compute_rmse(out, target, input)
        re = compute_re(out, target, input)

        # Log metrics
        self.log(f'{phase}_rmse', rmse, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{phase}_re', re, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        # print(f"50 * loss {50 * loss} + 1000 * grad_loss {1000 * grad_loss}+ 1.0 * prior_cost {1.0 * prior_cost}")
        return training_loss, out

    def base_step(self, batch, phase=""):
        out = self(batch=batch)
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
        out = self(batch=batch)
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


class GradSolver(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_grad = lr_grad

        self._grad_norm = None

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        return batch.input.nan_to_num().detach().requires_grad_(True)

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
        

        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            if not self.training:
                state = self.prior_cost.forward_ae(state)
        return state


    
class DoubleConv(nn.Module):
    """
    A helper module that performs two consecutive convolutions,
    each followed by BatchNorm and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)


class SimpleUNet(nn.Module):
    """
    A simple UNet architecture that maps an input tensor to an output tensor of the same shape.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        features (tuple of int, optional): Feature sizes for the encoder blocks.
            Default is (64, 128), which results in a two–level encoder.
    """
    def __init__(self, in_channels, out_channels, features=(64, 128)):
        super(SimpleUNet, self).__init__()
        # Encoder
        self.encoder1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features[0], features[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[1], features[1] * 2)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(features[1] * 2, features[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features[1] * 2, features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features[0] * 2, features[0])
        
        # Final output conv to map to desired number of channels
        self.conv_final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):#should we normalize x here???
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x =  x / self._grad_norm

        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))
        
        # Decoder path
        dec2 = self.up2(bottleneck)
        # Ensure the sizes match (in case of odd input dimensions)
        if dec2.size()[2:] != enc2.size()[2:]:
            dec2 = F.interpolate(dec2, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))
        
        dec1 = self.up1(dec2)
        if dec1.size()[2:] != enc1.size()[2:]:
            dec1 = F.interpolate(dec1, size=enc1.size()[2:], mode='bilinear', align_corners=True)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))
        
        return self.conv_final(dec1)
    
    def reset_state(self, inp):
        """
        For a UNet, there is no state to reset, so we skip it here
        """
        self._grad_norm = None
        pass


class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.dropout = torch.nn.Dropout(dropout)
        self._state = []
        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x):
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x =  x / self._grad_norm
        # print(f"Input shape: {x.shape}")
        hidden, cell = self._state
        # print(f"hidden, cell shape: {hidden.shape, cell.shape}")
        x = self.dropout(x)
        # print(f"Shape after dropout: {x.shape}")
        x = self.down(x)
        # print(f"Shape after down: {x.shape}")
        gates = self.gates(torch.cat((x, hidden), 1))
        # print(f"gates Shape : {gates.shape}")

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        # print(f"cell_gate Shape after tanh: {cell_gate.shape}")

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        # print(f"hidden Shape: {hidden.shape}")

        self._state = hidden, cell
        out = self.conv_out(hidden)
        # print(f"out Shape after conv_out: {out.shape}")
        out = self.up(out)
        # print(f"out Shape after up: {out.shape}")
        return out


class BaseObsCost(nn.Module):
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w=w

    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return self.w * F.mse_loss(state[msk], batch.input.nan_to_num()[msk])


class BilinAEPriorCost(nn.Module):
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

    def forward_ae(self, x):
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

    def forward(self, state):
        return F.mse_loss(state, self.forward_ae(state))

