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

    # # #changes: print everything here 
    # def init_state(self, batch, x_init=None):
    #     # Print the type and contents of the batch
    #     print("Batch type:", type(batch))
    #     print("Number of elements in batch tuple:", len(batch))

    #     # Loop through each element in the tuple and print its type and shape
    #     for i, element in enumerate(batch):
    #         print(f"Element {i} type: {type(element)}")
    #         if torch.is_tensor(element):
    #             print(f"Element {i} shape: {element.shape}")
    #         else:
    #             print(f"Element {i} content: {element}")

    #     if x_init is not None:
    #         return x_init

    #     # Adjust this line based on your batch structure
    #     return batch.input.nan_to_num().detach().requires_grad_(True)

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
        hidden, cell = self._state
        x = self.dropout(x)
        x = self.down(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        out = self.conv_out(hidden)
        out = self.up(out)
        return out


class BaseObsCost(nn.Module):
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w=w

    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return self.w * F.mse_loss(state[msk], batch.input.nan_to_num()[msk])

    
# class BilinAEPriorCost(nn.Module):##########Unet3D architecture, so bad!!! See report on Google Drive!
#     def __init__(self, dim_in=1, out_channels=1, dim_hidden=4, kernel_size=3,downsamp=None, bilin_quad=True):#base_filters=dim_hidden=4 corresp 139K paras, base_filters=8 corresp 202K paras and out of memories
#         super(BilinAEPriorCost, self).__init__()
        
#         # Encoder
#         self.conv_in = nn.Conv3d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2)#e11
#         self.conv_hidden = nn.Conv3d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2)#e12

#         self.conv_hidden = nn.Conv3d(
#             dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
#         )
#         self.bilin_1 = nn.Conv3d(
#             dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
#         )
#         self.bilin_21 = nn.Conv3d(
#             dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
#         )
#         self.bilin_22 = nn.Conv3d(
#             dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
#         )

#         self.conv_out = nn.Conv3d(
#             2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
#         )

#         self.down = nn.AvgPool3d(downsamp) if downsamp is not None else nn.Identity()
#         self.up = nn.Upsample(scale_factor=downsamp, mode='trilinear', align_corners=True) if downsamp is not None else nn.Identity()

#     def forward_ae(self, x):
#         print(f"Input shape: {x.shape}")
#         x = x.unsqueeze(1)
#         print(f"Shape after unsqueeze: {x.shape}")

#         # Encoder
#         print(f"Shape input: {x.shape}")
#         x = self.down(x)
#         print(f"After Down : {x.shape}")
#         x = self.conv_in(x)

#         print(f"After conv_in : {x.shape}")
#         x = self.conv_hidden(F.relu(x))

#         print(f"After conv_hidden : {x.shape}")

#         nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
#         print(f"After nonlin : {nonlin.shape}")
#         x = self.conv_out(
#             torch.cat([self.bilin_1(x), nonlin], dim=1)
#         )
#         print(f"After cat : {x.shape}")
#         out = self.up(x)
#         print(f"After up : {x.shape}")


#         out = out.squeeze(1)  # Remove the channel dimension after UNet
#         # print(f"Output shape after squeeze: {out.shape}")

#         return out


#     def forward(self, state):
#         out = self.forward_ae(state)
#         return F.mse_loss(state, out)
    
class BilinAEPriorCost(nn.Module):##########Unet3D architecture
    def __init__(self, in_channels=1, out_channels=1, dim_hidden=32, downsamp=None, bilin_quad=True):#base_filters=4 corresp 139K paras, base_filters=8 corresp 202K paras and out of memories
        super(BilinAEPriorCost, self).__init__()
        base_filters=dim_hidden
        # Encoder
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.e21 = nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1)
        self.e22 = nn.Conv3d(base_filters, base_filters, kernel_size=3, padding=1)
        
        # Output layer
        self.outconv_0 =nn.Conv3d(base_filters*2, base_filters, kernel_size=3, padding=3 // 2)
        # self.outconv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose3d(base_filters, out_channels, kernel_size=2, stride=2)

        #bilin
        self.bilin_1 = nn.Conv3d(
            base_filters, base_filters, kernel_size=3, padding=3 // 2)

        self.bilin_21 = nn.Conv3d(
            base_filters, base_filters, kernel_size=3, padding=3 // 2)
        self.bilin_22 = nn.Conv3d(
            base_filters, base_filters, kernel_size=3, padding=3 // 2)


    def forward_ae(self, x):
        # print(f"Input shape: {x.shape}")
        x = x.unsqueeze(1)
        # print(f"Shape after unsqueeze: {x.shape}")
        xp1 = self.pool1(x)
        # print(f"Shape after pool1: {xp1.shape}")
        xe21 = nn.ReLU(inplace=True)(self.e21(xp1))
        # print(f"Shape after e21 and ReLU: {xe21.shape}")
        xe22 = nn.ReLU(inplace=True)(self.e22(xe21))
        # print(f"Shape after e22 and ReLU: {xe22.shape}")
        nonlin = self.bilin_21(xe22) * self.bilin_22(xe22)
        # print(f"Shape after nonlin: {nonlin.shape}")
        x33 = self.outconv_0(torch.cat([self.bilin_1(xe22), nonlin], dim=1))
        # print(f"Shape after x33: {x33.shape}")
        xu1 = self.upconv1(x33)
        # print(f"Shape after xu1: {xu1.shape}")
        out = xu1.squeeze(1)  # Remove the channel dimension after UNet
        # print(f"Output shape after squeeze: {out.shape}")
        return out
    
    def forward(self, state):
        out = self.forward_ae(state)
        return F.mse_loss(state, out)
    
