from cmath import phase
from collections import namedtuple
import functools as ft
import time
import numpy as np
import torch
import kornia.filters as kfilts
import xarray as xr



#Nga comment out unused imports, instead I copied it in the src folder
# from ocean4dvarnet.data import BaseDataModule, TrainingItem
from .data import BaseDataModule, TrainingItem
from .models import Lit4dVarNet,GradSolver
# from ocean4dvarnet.models import Lit4dVarNet,GradSolver


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


class GradModelWithCondition(torch.nn.Module):
    """
    A generic conditional model for gradient modulation.

    Attributes:
        grad_model : grad update model
    """

    def __init__(self, grad_model=False, dropout=0.,use_grad_norm=True):
        """
        Initialize the ConvLstmGradModel.

        Args:
            grad_model : grad update model
        """
        super().__init__()
        self.grad_model = grad_model
        self.dropout = torch.nn.Dropout(dropout)
        self.use_grad_norm = use_grad_norm

        if hasattr(self.grad_model, 'dim_3d') == True:
            self.dim_3d = self.grad_model.dim_3d

        if hasattr(self.grad_model, 'dims') == True:
            if self.grad_model.dims == 3:
                self.dim_3d = True

    def reset_state(self, inp):
        """
        Reset the internal state of the LSTM.

        Args:
            inp (torch.Tensor): Input tensor to determine state size.
        """
        self._grad_norm = None


    def forward(self, x, timesteps=None, extra=[]):
        """
        Perform the forward pass of the LSTM.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        if self._grad_norm is None:
            if self.use_grad_norm:
                self._grad_norm = (x**2).mean().sqrt()
            else:
                self._grad_norm = 1.

        #print('self._grad_norm in GradModelWithCondition:', self._grad_norm, flush=True)
        x = x / self._grad_norm

        x = self.dropout(x)
        out = self.grad_model.predict(x, timesteps=timesteps, extra=extra)
        
        # print("x rms:", torch.sqrt((x**2).mean()).item(),
              
        #       "out rms:", torch.sqrt((out**2).mean()).item(),
              
        # "t min/max:", timesteps.min().item(), timesteps.max().item(),
        # "x shape:", tuple(x.shape),
        # "t shape:", tuple(timesteps.shape),
        # flush=True)

        return out

class GradSolver_withStep(GradSolver):

    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, lbd=1.0, **kwargs):
        """
        Initialize the GradSolver.

        Args:
            prior_cost (nn.Module): The prior cost function.
            obs_cost (nn.Module): The observation cost function.
            grad_mod (nn.Module): The gradient modulation model.
            n_step (int): Number of optimization steps.
            lr_grad (float, optional): Learning rate for gradient updates. Defaults to 0.2.
            lbd (float, optional): Regularization parameter. Defaults to 1.0.
        """
        self.input_grad_update = kwargs.pop(
            "input_grad_update",
            kwargs["input_grad_update"],
        )
        print("Input type for GradSolver =",self.input_grad_update,flush=True)
        std_init = kwargs.pop( "std_init",None)
        print("Std init for GradSolver =",std_init,flush=True)

        super().__init__(prior_cost, obs_cost, grad_mod, n_step=n_step, lr_grad=lr_grad, lbd=lbd,**kwargs)

        self.grad_mod._grad_norm = None
        self.h_state = None

        if std_init is not None:
            self.std_init = std_init

    def init_state(self, batch, x_init=None):
        """
        Initialize the state for optimization.

        Args:
            batch (dict): Input batch containing data.
            x_init (torch.Tensor, optional): Initial state. Defaults to None.
        Returns:
            torch.Tensor: Initialized state.
        """
        if x_init is not None:
            return x_init.detach().requires_grad_(True)

        if hasattr(self, 'std_init') is True :
            x0 = self.std_init * torch.randn_like(batch.input)
            return x0.detach().requires_grad_(True)
        else:
            return torch.zeros_like(batch.input).detach().requires_grad_(True)

    def init_h_state(self, batch, h_state=None):
        """
        Initialize the state for optimization.

        Args:
            batch (dict): Input batch containing data.
            x_init (torch.Tensor, optional): Initial state. Defaults to None.

        Returns:
            torch.Tensor: Initialized state.
        """
        if h_state is not None:
            self.h_state = h_state
        else:
            self.h_state = torch.zeros_like(batch.input).detach().requires_grad_(True)

    def format2D_3D(self, x):
        if hasattr(self.grad_mod, 'dim_3d') == True:
            if self.grad_mod.dim_3d == True:
                x =  x.unsqueeze(1)

        return x

    def solver_step(self, state, batch, step, alpha_step=1.):
        """
        Perform a single optimization step.

        Args:
            state (torch.Tensor): Current state.
            batch (dict): Input batch containing data.
            step (float): Current optimization step between 0 and 1.
            alpha_step (float): scaling factor for the step.

        Returns:
            torch.Tensor: Updated state.
        """
    
        if( isinstance(step, float) ):
            t = torch.tensor([step], device=state.device).repeat(state.shape[0])
        else:
            t = step
        #print(t)    

        if 'subgrad' in self.input_grad_update :
            gobs = (batch.input-state).nan_to_num()
            gprior = state - self.prior_cost.forward_ae(state)
            grad = torch.concatenate((self.format2D_3D(gobs),self.format2D_3D(gprior)),dim=1)

            if 'state' in self.input_grad_update :
                grad = torch.concatenate((grad,self.format2D_3D(state)),dim=1)

            if 'previous' in self.input_grad_update :
                grad = torch.concatenate((grad,self.format2D_3D(self.h_state)),dim=1)

        elif 'grad' in self.input_grad_update :
            var_cost = self.prior_cost(state) + self.lbd**2 * self.obs_cost(state, batch)
            grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

            #if self.grad_mod._grad_norm is None:
            #    self.grad_mod._grad_norm = (grad**2).mean().sqrt().detach()
            #grad = grad / self.grad_mod._grad_norm

            if 'state' in self.input_grad_update :
                grad = torch.concatenate((  self.format2D_3D(grad),self.format2D_3D(state)),dim=1)

            if 'previous' in self.input_grad_update :
                grad = torch.concatenate((grad,self.format2D_3D(self.h_state)),dim=1)

        elif  self.input_grad_update == 'obs-only' :
            grad = batch.input.nan_to_num()

        elif  self.input_grad_update == 'obs+state' :
            grad = torch.concatenate((self.format2D_3D(state),self.format2D_3D(batch.input.nan_to_num())),dim=1)

        gmod = self.grad_mod(grad, timesteps=t, extra=None)
        if hasattr(self.grad_mod, 'dim_3d') == True:
            if self.grad_mod.dim_3d == True:
                gmod = gmod.squeeze(1)

        state_update = alpha_step * gmod
        if ( 'grad' in self.input_grad_update ) and ( self.lr_grad > 0. ) : 
            state_update += self.lr_grad * (step + 1) / self.n_step * grad[:,:state.shape[1],:,:]

        self.h_state = state_update
        
        # if torch.rand(1).item() < 0.1:  # occasionally
        #     print("gobs rms:", torch.sqrt((gobs**2).mean()).item(),
        #         "gprior rms:", torch.sqrt((gprior**2).mean()).item(),
        #         "state rms:", torch.sqrt((state**2).mean()).item(),
        #         "gmod rms:", torch.sqrt((gmod**2).mean()).item(),
        #         flush=True)
    

        return state - state_update
        
    def forward(self, batch, x_init=None, h_state=None,phase='test'):
        """
        Perform the forward pass of the solver.

        Args:
            batch (dict): Input batch containing data.

        Returns:
            torch.Tensor: Final optimized state.
        """
        with torch.set_grad_enabled(True):
            state = self.init_state(batch, x_init=x_init)
            self.init_h_state(batch, h_state=h_state)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):

                alpha_step = 1. / self.n_step               
                state = self.solver_step(state, batch, step= step / self.n_step, alpha_step=alpha_step)
                if ( not self.training ) and ( 'grad' in self.input_grad_update ):
                    state = state.detach().requires_grad_(True)

        return state


### UNET SOLVER

# --- Bloc de base ResNet ---
class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim, dropout=0.0, bias=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias)
        self.gn1 = torch.nn.GroupNorm(max(1, out_ch // 8), out_ch)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias)
        self.gn2 = torch.nn.GroupNorm(max(1, out_ch // 8), out_ch)

        self.act = lambda x: x * torch.sigmoid(x)  # Swish
        self.dense = Dense(embed_dim, out_ch)      # time embedding projection
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        # skip 1x1 conv si dimensions changent
        self.skip = torch.nn.Conv2d(in_ch, out_ch, 1, bias=bias) if in_ch != out_ch else torch.nn.Identity()

    def forward(self, x, embed):
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h + self.dense(embed))
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h + self.dense(embed))

        return h + self.skip(x)


class GaussianFourierProjection(torch.nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = torch.nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    ret = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    return ret

class Dense(torch.nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = torch.nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]



class UnetGradModelUnet(torch.nn.Module):
    """Score-based UNet avec ResNet blocks et time embedding."""

    def __init__(self, dim_in, dim_hidden, embed_dim, num_levels, unet, out_activation=None, bias=False, dropout=0.0):
        super().__init__()

        # progression des canaux
        channels = [dim_hidden]
        for i in range(num_levels - 1):
            channels.append(channels[-1] * 2)

        # time embedding
        self.embed = torch.nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            torch.nn.Linear(embed_dim, embed_dim)
        )
        self.act = lambda x: x * torch.sigmoid(x)
        self.norm = torch.nn.Parameter(torch.tensor([1.]))

        # --- Encoding ---
        self.enc_blocks = torch.nn.ModuleList()
        in_ch = dim_in
        for ch in channels:
            self.enc_blocks.append(ResBlock(in_ch, ch, embed_dim, bias=bias, dropout=dropout))
            in_ch = ch

        # --- Bottleneck ---
        self.bottleneck = ResBlock(channels[-1], channels[-1], embed_dim, dropout=dropout)

        # --- Decoding ---
        self.dec_blocks = torch.nn.ModuleList()
        for i in reversed(range(1, len(channels))):
            self.dec_blocks.append(
                torch.nn.ModuleDict({
                    "upsample": torch.nn.ConvTranspose2d(channels[i], channels[i-1], 4, stride=2, padding=1, bias=bias),
                    "resblock": ResBlock(channels[i-1]*2, channels[i-1], embed_dim, dropout=dropout, bias=bias)
                })
            )

        # --- Final ---
        if unet is not None:
            self.conv_out = unet
            self.use_unet = True
        else:
            self.use_unet = False
            self.conv_out = torch.nn.Conv2d(channels[0]*2, dim_in, 3, padding=1)


        # Option activation de sortie
        if out_activation == "tanh":
            self.out_act = torch.nn.Tanh()
        elif out_activation == "sigmoid":
            self.out_act = torch.nn.Sigmoid()
        else:
            self.out_act = torch.nn.Identity()

    def reset_state(self, inp):
        self._grad_norm = None

    def forward(self, x, t):
        if self._grad_norm is None:
            self._grad_norm = (x ** 2).mean().sqrt()
        x = x / self._grad_norm

        # time embedding 
        embed = self.act(self.embed(t))

        # --- Encoder ---
        hs = []
        h = x
        for block in self.enc_blocks:
            h = block(h, embed)
            hs.append(h)
            h = torch.nn.functional.avg_pool2d(h, 2) if block != self.enc_blocks[-1] else h  # downsample sauf dernier

        # --- Bottleneck ---
        h = self.bottleneck(h, embed)

        # --- Decoder ---
        skip_connections = hs[::-1]
        for skip, dec in zip(skip_connections[1:], self.dec_blocks):  # on garde aussi skip du niveau le + bas
            h = dec["upsample"](h)
            h = dec["resblock"](torch.cat([h, skip], dim=1), embed)

        # --- Final ---
        if self.use_unet == True:
            out = self.conv_out.predict(torch.cat([h, skip_connections[-1]], dim=1))
        else:
            out = self.conv_out(torch.cat([h, skip_connections[-1]], dim=1))

        return self.out_act(out)

class Lit4dVarNetIgnoreNaN(Lit4dVarNet):
    def __init__(self,  
                 w_mse,w_grad_mse, w_mse_lr, w_grad_mse_lr, w_prior,
                 *args, **kwargs):
        _val_rec_weight = kwargs.pop(
            "val_rec_weight",
            kwargs["rec_weight"],
        )

        self.osse_with_interp_error = kwargs.pop("osse_with_interp_error",False)

        print('osse_with_interp_error:', self.osse_with_interp_error)

        super().__init__(*args, **kwargs)

        self.register_buffer(
            "val_rec_weight",
            torch.from_numpy(_val_rec_weight),
            persistent=False,
        )

        self._n_rejected_batches = 0

        self.w_mse = w_mse
        self.w_grad_mse = w_grad_mse
        self.w_mse_lr = w_mse_lr
        self.w_grad_mse_lr = w_grad_mse_lr
        self.w_prior = w_prior

    def get_rec_weight(self, phase):
        rec_weight = self.rec_weight
        if phase == "val":
            rec_weight = self.val_rec_weight
        return rec_weight

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if loss is None:
            self._n_rejected_batches += 1
        return loss

    # def configure_optimizers(self):
    #     print('run this configure_optimizers')
    #     opt = torch.optim.Adam(self.parameters(), lr=1e-4)
    #     return opt

    # def on_after_backward(self):
    #     gm = self.solver.grad_mod.grad_model  # UNetModel2 instance
    #     last = gm.out[-1]
    #     if last.weight.grad is None:
    #         print("[after_backward] last conv grad: None")
    #     else:
    #         print("[after_backward] last conv grad rms:", last.weight.grad.detach().pow(2).mean().sqrt().item())


    def on_train_epoch_end(self):
        self.log(
            "n_rejected_batches",
            self._n_rejected_batches,
            on_step=False,
            on_epoch=True,
        )

    def sample_osse_data_with_l3interp_errr(self,batch):
        # to be implemented in child class if needed
        # patch dimensions
        K = batch.input.shape[0]
        N = batch.input.shape[2]
        M = batch.input.shape[3]
        T = batch.input.shape[1]

        # start with time interpolation error only
        dt =  1. * ( torch.rand((K,T-2,N,M), dtype=torch.float32, device=batch.input.device) - 0.5 )
        dt = torch.nn.functional.avg_pool2d(dt,(4,4))
        dt = torch.nn.functional.interpolate(dt,scale_factor=4.,mode='bilinear')
          
        inp_dt = (dt >= 0. ) * ( (1.-dt) * batch.tgt[:,1:-1,:,:] + dt * batch.tgt[:,2:,:,:] )
        inp_dt += ( dt < 0.) * ( (1+dt) * batch.tgt[:,1:-1,:,:] - dt * batch.tgt[:,0:-2,:,:] )
        inp_dt = torch.cat( ( batch.tgt[:,0:1,:,:], inp_dt, batch.tgt[:,-1:,:,:] ), dim = 1)

        #print('\n ....... interpolation time' )
        #print(batch.tgt[0,0:5,10,10].detach().cpu().numpy(),inp_dt[0,1,10,10].detach().cpu().numpy() , dt[0,0,10,10].detach().cpu().numpy()     )

        # space interpolation error
        scale_spatial_perturbation = 1.
        dx = scale_spatial_perturbation * torch.rand((K,T,N,M), dtype=torch.float32, device=batch.input.device)
        dy = scale_spatial_perturbation * torch.rand((K,T,N,M), dtype=torch.float32, device=batch.input.device)

        dx = torch.nn.functional.avg_pool2d(dx,(4,4))
        dx = torch.nn.functional.interpolate(dx,scale_factor=4.,mode='bilinear')

        dy = torch.nn.functional.avg_pool2d(dy,(4,4))
        dy = torch.nn.functional.interpolate(dy,scale_factor=4.,mode='bilinear')

        dx = dx[:,:,:-1,:-1]
        dy = dy[:,:,:-1,:-1]

        inp_dxdydt = inp_dt[:,:,:-1,:-1] * (1-dx) * (1-dy) 
        inp_dxdydt += inp_dt[:,:,:-1,1:] * (1-dx) * dy
        inp_dxdydt += inp_dt[:,:,1:,:-1] * dx * (1-dy)
        inp_dxdydt += inp_dt[:,:,1:,1:] * dx * dy

        inp_dxdydt = torch.where( inp_dxdydt.isfinite() , inp_dxdydt, batch.tgt[:,:,:-1,:-1] )

        #print('\n....... interpolation space' )
        #print(inp_dt[0,2,10,10].detach().cpu().numpy(),inp_dxdydt[0,2,10,10].detach().cpu().numpy(), dx[0,2,10,10].detach().cpu().numpy(), dy[0,2,10,10].detach().cpu().numpy() )


        inp_dxdydt = torch.cat( ( inp_dxdydt, inp_dt[:,:,-1:,:-1] ), dim = 2)
        inp_dxdydt = torch.cat( ( inp_dxdydt, inp_dt[:,:,:,-1:] ), dim = 3)

        input = torch.where( batch.input.isfinite() ,  batch.input + inp_dxdydt.detach() - batch.tgt , torch.nan )
        #input = torch.where( batch.input.isfinite() ,  inp_dxdydt.detach() , torch.nan )
        input = input.detach()

        display = None #True #
        if display is not None:
            noise = ( input - batch.tgt ) * self.norm_stats[1]#self.norm_stats['train'][1]
            print("..... mean, std of the simulated spatial perturnation noise : %.3f -- %.3f "%(torch.nanmean(noise), torch.sqrt( torch.nanmean(noise**2) - torch.nanmean(noise)**2) ))
            #print("..... number of observed pixels (new) : %.2f  "%(100 * input.isfinite().float().mean()) )
            #print("..... number of observed pixels (new) : %.2f  "%(100 * noise.isfinite().float().mean()) )
            #print("..... number of observed pixels (orig): %.2f "%(100 * batch.input.isfinite().float().mean()) )


        return TrainingItem(input, batch.tgt)


    def loss_mse(self,batch,out,phase):
        loss =  self.weighted_mse(out - batch.tgt,
            self.get_rec_weight(phase),
        )

        grad_loss =  self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        return loss, grad_loss

    def loss_prior(self,batch,out,phase):

        # prior cost for estimated latent state    
        loss_prior_out = self.solver.prior_cost(out) # Why using init_state

        # prior cost for true state
        loss_prior_tgt = self.solver.prior_cost( batch.tgt.nan_to_num() )

        return loss_prior_out,loss_prior_tgt

    def step(self, batch, phase):

        # if self.training and batch.tgt.isfinite().float().mean() < 0.5:
        #     return None, None

        # osse input
        #print( 'sampling osse data with l3 interp error: ', self.osse_with_interp_error , flush=True)
        #print('... phase: ', phase , flush=True)
        if ( self.osse_with_interp_error == True ) and ( ( phase == "train" ) or ( phase == "val" ) ):
            
            batch_ = self.sample_osse_data_with_l3interp_errr(batch)
        else:
            #print('... raw batch')
            batch_ = batch
        
        # apply base-step
        loss, out = self.base_step(batch_, phase)

        loss_mse = self.loss_mse(batch,out,phase)
        loss_prior = self.loss_prior(batch,out.detach(),phase)

        training_loss = self.w_mse * loss_mse[0] + self.w_grad_mse * loss_mse[1]
        training_loss += self.w_prior * loss_prior[0] + self.w_prior * loss_prior[1]
        
        # Compute custom metrics with masking
        input=batch.input
        target=batch.tgt
        rmse = compute_rmse(out, target, input)
        re = compute_re(out, target, input)

        # Log metrics

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss_mse[0] * self.norm_stats[1] ** 2, #self.norm_stats[phase][1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_loss",
                training_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            
            self.log(f'{phase}_rmse', rmse, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'{phase}_re', re, prog_bar=True, on_step=False, on_epoch=True)
            
            self.log(
                f"{phase}_gloss",
                loss_mse[1],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_ploss_out",
                loss_prior[0],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_ploss_gt",
                loss_prior[1],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

        return training_loss, out

    def base_step(self, batch, phase):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))

        return loss, out
    
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     if batch_idx == 0:
    #         gm = self.solver.grad_mod.grad_model
    #         last = gm.out[-1]
    #         print("[after_step] last conv |w| mean:", last.weight.detach().abs().mean().item())


    def forward(self, batch):
        """
        Forward pass through the solver.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: Solver output.
        """
        return self.solver(batch)
 

class UnetSolver(torch.nn.Module):
    def __init__(self, dim_in, channel_dims, max_depth=None,bias=True):
        super().__init__()

        if max_depth is not None :
            self.max_depth = np.max( max_depth , len(channel_dims) // 3 )
        else: 
            self.max_depth = len(channel_dims) // 3
        
        self.ups = torch.nn.ModuleList()
        self.up_pools = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.residues = list()

        self.bottom_transform = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3 - 1],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias
            ),
            torch.nn.ReLU(),
        )

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=dim_in,
                padding="same",
                kernel_size=3,
                bias=bias
            )
        )

        self.final_linear = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_in))

        for depth in range(self.max_depth):
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 2] * 2,
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                )
            )
            self.up_pools.append(
                torch.nn.ConvTranspose2d(
                    in_channels=channel_dims[depth * 3 + 3],
                    out_channels=channel_dims[depth * 3 + 2],
                    kernel_size=2,
                    stride=2,
                    bias=bias
                )
            )
            self.downs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=dim_in
                        if depth == 0
                        else channel_dims[depth * 3 - 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3],
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                )
            )
            self.down_pools.append(torch.nn.MaxPool2d(kernel_size=2))

    def unet_step(self, x, depth):
        x, residue = self.down(x, depth)
        self.residues.append(residue)

        if depth == self.max_depth - 1:
            x = self.bottom_transform(x)
        else:
            x = self.unet_step(x, depth + 1)

        return self.up(x, depth)

    def forward(self, batch):
        x = batch.input
        x = x.nan_to_num()
 #       x = self.final_up(self.unet_step(x, depth=0))
 #       x = torch.permute(x, dims=(0, 2, 3, 1))
 #       x = self.final_linear(x)
 #       x = torch.permute(x, dims=(0, 3, 1, 2))
        return self.predict(x)

    def predict(self,x):
        x = self.final_up(self.unet_step(x, depth=0))
        x = torch.permute(x, dims=(0, 2, 3, 1))
        x = self.final_linear(x)
        x = torch.permute(x, dims=(0, 3, 1, 2))
        return x        

    def down(self, x, depth):
        x = self.downs[depth](x)
        return self.down_pools[depth](x), x

    def up(self, x, depth):
        x = self.up_pools[depth](x)
        x = self.concat_residue(x)
        return self.ups[depth](x)

    def concat_residue(self, x):
        if len(self.residues) != 0:
            residue = self.residues.pop(-1)

            _, _, h_x, w_x = x.shape
            _, _, h_r, w_r = residue.shape

            pad_h = h_r - h_x
            pad_w = w_r - w_x

            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect", value=0)

            return torch.concat((x, residue), dim=1)
        else:
            return x



class GenericAEPriorCost(torch.nn.Module):
    """
    A prior cost model using bilinear autoencoders.

    Attributes:
        bilin_quad (bool): Whether to use bilinear quadratic terms.
        conv_in (nn.Conv2d): Convolutional layer for input.
        conv_hidden (nn.Conv2d): Convolutional layer for hidden states.
        bilin_1 (nn.Conv2d): Bilinear layer 1.
        bilin_21 (nn.Conv2d): Bilinear layer 2 (part 1).
        bilin_22 (nn.Conv2d): Bilinear layer 2 (part 2).
        conv_out (nn.Conv2d): Convolutional layer for output.
        down (nn.Module): Downsampling layer.
        up (nn.Module): Upsampling layer.
    """

    def __init__(self, model_ae):
        """
        Initialize the BilinAEPriorCost module.

        Args:
            dim_in (int): Number of input dimensions.
            dim_hidden (int): Number of hidden dimensions.
            kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
            downsamp (int, optional): Downsampling factor. Defaults to None.
            bilin_quad (bool, optional): Whether to use bilinear quadratic terms. Defaults to True.
        """
        super().__init__()

        self.model_ae = model_ae 

    def forward_ae(self, x):
        """
        Perform the forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the autoencoder.
        """
        return self.model_ae(x)

    def forward(self, state):
        """
        Compute the prior cost using the autoencoder.

        Args:
            state (torch.Tensor): The current state tensor.

        Returns:
            torch.Tensor: The computed prior cost.
        """
        return torch.nn.functional.mse_loss(state, self.forward_ae(state))


class UnetSolver2(UnetSolver):
    def __init__(self, dim_in, channel_dims, max_depth=None,dim_out=None,bias=True):
        super().__init__(dim_in, channel_dims, max_depth)

        if dim_out is None :
            dim_out = dim_in

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=4*dim_out,
                padding="same",
                kernel_size=3,
                bias=bias
            ) )

        self.final_linear = torch.nn.Sequential(torch.nn.Linear(4*dim_out, dim_out))


class UpsampleWInterpolate(torch.nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, out_channels=None, interp_mode='bilinear',bias=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.interp_mode = interp_mode
        if use_conv:
            self.conv  = torch.nn.Conv2d(in_channels=channels,out_channels=out_channels,
                                        padding="same",kernel_size=1,bias=bias)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode=self.interp_mode)
        if self.use_conv:
            x = self.conv(x)
        return x
