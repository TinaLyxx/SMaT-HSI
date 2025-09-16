import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from einops import rearrange, repeat

from functools import partial
from typing import Callable
from timm.models.layers import DropPath, to_2tuple


from utils.pos_embed import get_2d_sincos_pos_embed
from utils.selective_scan import selective_scan_state_flop_jit, selective_scan_fn
from utils.dwconv_layer import DepthwiseFunction



class StateFusion(nn.Module):
    def __init__(self, dim):
        super(StateFusion, self).__init__()

        self.dim = dim
        self.kernel_3   = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):
        h = h.contiguous()
        if self.training:
            h1 = F.conv2d(self.padding(h, (1,1,1,1)), self.kernel_3,   padding=0, dilation=1, groups=self.dim)
            h2 = F.conv2d(self.padding(h, (3,3,3,3)), self.kernel_3_1, padding=0, dilation=3, groups=self.dim)
            h3 = F.conv2d(self.padding(h, (5,5,5,5)), self.kernel_3_2, padding=0, dilation=5, groups=self.dim)
            out = self.alpha[0]*h1 + self.alpha[1]*h2 + self.alpha[2]*h3
            return out

        else:
            if not hasattr(self, "_merge_weight"):
                self._merge_weight = torch.zeros((self.dim, 1, 11, 11), device=h.device)
                # self._merge_weight[:, :, 2:5, 2:5] = self.alpha[0]*self.kernel_3

                # self._merge_weight[:, :, 0:1, 0:1] = self.alpha[1]*self.kernel_3_1[:,:,0:1,0:1]
                # self._merge_weight[:, :, 0:1, 3:4] = self.alpha[1]*self.kernel_3_1[:,:,0:1,1:2]
                # self._merge_weight[:, :, 0:1, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,0:1,2:3]
                # self._merge_weight[:, :, 3:4, 0:1] = self.alpha[1]*self.kernel_3_1[:,:,1:2,0:1]
                # self._merge_weight[:, :, 3:4, 3:4] += self.alpha[1]*self.kernel_3_1[:,:,1:2,1:2]
                # self._merge_weight[:, :, 3:4, 6:7] = self.alpha[1]*self.kernel_3_1[:,:,1:2,2:3]
                # self._merge_weight[:, :, 6:7, 0:1] = self.alpha[1]*self.kernel_3_1[:,:,2:3,0:1]
                # self._merge_weight[:, :, 6:7, 3:4] = self.alpha[1]*self.kernel_3_1[:,:,2:3,1:2]
                # self._merge_weight[:, :, 6:7, 6:7] = self.alpha[1]*self.kernel_3_1[:,:,2:3,2:3]

                # 11*11
                self._merge_weight[:, :, 4:7, 4:7] = self.alpha[0]*self.kernel_3

                self._merge_weight[:, :, 2:3, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,0:1,0:1]
                self._merge_weight[:, :, 2:3, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,0:1,1:2]
                self._merge_weight[:, :, 2:3, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[1]*self.kernel_3_1[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,1:2,2:3]
                self._merge_weight[:, :, 8:9, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,2:3,0:1]
                self._merge_weight[:, :, 8:9, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,2:3,1:2]
                self._merge_weight[:, :, 8:9, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,2:3,2:3]

                self._merge_weight[:, :, 0:1, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,0:1,0:1]
                self._merge_weight[:, :, 0:1, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,0:1,1:2]
                self._merge_weight[:, :, 0:1, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[2]*self.kernel_3_2[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,1:2,2:3]
                self._merge_weight[:, :, 10:11, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,2:3,0:1]
                self._merge_weight[:, :, 10:11, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,2:3,1:2]
                self._merge_weight[:, :, 10:11, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,2:3,2:3]


            out = DepthwiseFunction.apply(h, self._merge_weight, None, 11//2, 11//2, False)

            return out
        

class StructureAwareSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        spe_patch = 5, # changable
        conv_bias=True,
        bias=False,
        spe = False,
        cls = True,
        device=None,
        dtype=None,
        use_fusion=True,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn
        self.use_fusion = use_fusion
        if use_fusion:
            if not spe:
                self.state_fusion = StateFusion(self.d_inner)
            else:
                self.spe_fusion = nn.Conv1d(
                    in_channels=self.d_inner, 
                    out_channels=self.d_inner,
                    kernel_size=spe_patch,
                    padding=(spe_patch - 1) // 2,
                    groups=self.d_inner)
        else:
            self.state_fusion = self.spe_fusion = None

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.spe = spe
        self.cls = cls


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj
 
    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D) 
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, xs: torch.Tensor):
        B, C, L = xs.shape
        H = W = int(L**0.5)
        
        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)
        
        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        h = self.selective_scan(
            xs, dts, 
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        if self.use_fusion:
            if self.cls:
                cls_token = h[:,:,:,-1:]
                cls_token = rearrange(cls_token, "b d c l -> b (d c) l")
                h = h[:,:,:,:-1]
                
            if self.spe:
                h = rearrange(h, "b d c l -> b (d c) l")
                h = self.spe_fusion(h)
            else:
                h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
                h = self.state_fusion(h)
                h = rearrange(h, "b d h w -> b d (h w)")
            
            if self.cls:
                h = torch.cat((h, cls_token), dim=2)
        else:
            h = rearrange(h, "b d c l -> b (d c) l")
        
        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, L, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.contiguous().chunk(2, dim=-1) 


        x = x.permute(0, 2, 1)

        # x = rearrange(x, 'b h w d -> b d h w').contiguous()
        # x = self.act(self.conv2d(x)) 

        y = self.ssm(x) 

        y = rearrange(y, 'b d l-> b l d')

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y
    

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class block_1D(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 1, # 16
        dt_init: str = "random",
        bi: bool = True,
        cls: bool = True,
        spe: bool = False,
        transformer_layer: list = [3],
        counter: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        if counter in transformer_layer:
            self.self_attention = Attention(
                hidden_dim,
                num_heads=8,
                qkv_bias=True,
                qk_norm=None,
                attn_drop=0.,
                proj_drop=0.,
                norm_layer=norm_layer,
            )
        else:
            self.self_attention = StructureAwareSSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, dt_init=dt_init, spe=spe, cls=cls,**kwargs)
            # self.self_attention = SSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.bi = bi
        self.cls = cls

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x1 = self.self_attention(x)
        if self.bi:
            if self.cls:
                x2 = x[:,0:-1,:]
                cls_token = x[:,-1:,:]
                x2 = torch.flip(x2, dims=[1])
                x2 = torch.cat((x2, cls_token), dim=1)
                x3 = self.self_attention(x2)

                x2 = x3[:,0:-1,:]
                cls_token = x3[:,-1:,:]
                x3 = torch.flip(x2, dims=[1])
                x3 = torch.cat((x3, cls_token), dim=1)
            else:
                x3 = torch.flip(x, dims=[1])
                x3 = self.self_attention(x3)
                x3 = torch.flip(x3, dims=[1])
            return self.drop_path((x1+x3)/2) + input
        else:
            return self.drop_path(x1) + input



class PatchEmbed_2D(nn.Module):
    """ 2D image to Patch Embedding
    """
    def __init__(self, img_size=(224,224), patch_size=16, in_chans=3, embed_dim=64, norm_layer=None, flatten = True):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            # x = spiral_flatten(x).transpose(1, 2)
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            # x = s_flatten(x).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchEmbed_Spe(nn.Module):
    """ 2D image to Patch Embedding
    """
    def __init__(self, img_size=(9,9), patch_size=2, embed_dim=64, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv1d(
            in_channels=img_size[0]*img_size[1],
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)
        x = x.transpose(2,1) # B,L,C
        x = self.proj(x)
        x = x.transpose(2,1)
        x = self.norm(x)
        return x


class spectral_spatial_block(nn.Module):
    def __init__(self, embed_dim, bi=False, N=8, drop_path=0.0, norm_layer=nn.LayerNorm, counter=0, cls = True, fu = True):
        super(spectral_spatial_block, self).__init__()
        self.spa_block = block_1D(
            # This module uses roughly 3 * expand * d_model^2 parameters
            hidden_dim=embed_dim, # Model dimension d_model
            drop_path = drop_path,
            bi = bi,
            cls = cls,
            spe = False,
            transformer_layer = [3], # changable
            counter = counter
            # gaussian = True
            )
        self.spe_block = block_1D(
            # This module uses roughly 3 * expand * d_model^2 parameters
            hidden_dim=embed_dim, # Model dimension d_model
            drop_path = drop_path,
            bi = bi,
            cls = cls,
            spe = True,
            transformer_layer = [3], # changable
            counter = counter
            )
        # self.linear = nn.Linear(N, N)
        # self.norm = norm_layer(embed_dim)
        self.l1= nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias = False),
            nn.Sigmoid(),)
        self.fu = fu

    def forward(self, x_spa, x_spe):
        ###  x:(B, L, D)
        x_spa = self.spa_block(x_spa)   #(N, HW/P^2, D)
        B, N, D = x_spa.shape
        x_spe = self.spe_block(x_spe)   #(N, B, D)
        _,N1,_ = x_spe.shape

        if self.fu:
            x_spa_c = x_spa[:,(N-1)//2,:]
            
            x_spe_c = x_spe.mean(1)
            sig = self.l1((x_spa_c+x_spe_c)/2).unsqueeze(1)
            x_spa = x_spa*sig.expand(-1,N,-1)
            x_spe = x_spe*sig.expand(-1,N1,-1)

        return x_spa, x_spe

def positional_embedding_1d(seq_len, embed_size):
    position_enc = torch.zeros(seq_len, embed_size)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))
    position_enc[:, 0::2] = torch.sin(position.float() * div_term)
    position_enc[:, 1::2] = torch.cos(position.float() * div_term)
    return position_enc.unsqueeze(0)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=(224,224), patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = nn.Sequential(
            ConvLayer(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim, embed_dim * 4, kernel_size=patch_size, stride=patch_size, padding=0, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):

        x = self.conv1(x) + x
        x = self.conv2(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class mamba_spatial(nn.Module):
    def __init__(self, spa_img_size=(224, 224),spe_img_size=(5,5), spa_patch_size=16, spe_patch_size=2, in_chans=3, hid_chans = 32, embed_dim=128, nclass=10, drop_path=0.0, depth=4, bi=True, 
                 norm_layer=nn.LayerNorm, global_pool=True, cls = True, fu=True):
        super().__init__()

        self.dimen_redu = nn.Sequential(
            nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans),
            # nn.GroupNorm(4, hid_chans),
            nn.ReLU(),

            nn.Conv2d(hid_chans, hid_chans, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans),
            # nn.GroupNorm(4, hid_chans),
            # nn.SiLU(),
            )

        self.half_spa_patch_size = spa_img_size[0] // 2
        self.half_spe_patch_size = spe_img_size[0] // 2
        self.spe_patch_embed = PatchEmbed_Spe(img_size=spe_img_size, patch_size=spe_patch_size, embed_dim=embed_dim)
        self.spa_patch_embed = PatchEmbed_2D(spa_img_size, spa_patch_size, hid_chans, embed_dim)
        # self.spa_patch_embed = Stem(img_size=spa_img_size, patch_size=spa_patch_size, in_chans=hid_chans, embed_dim=embed_dim)
        spa_num_patches = self.spa_patch_embed.num_patches
        if in_chans % spe_patch_size ==0:
          spe_num_patches = in_chans//spe_patch_size
        else:
          spe_num_patches = in_chans//spe_patch_size

        self.cls = cls
        if self.cls:
          self.spa_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
          self.spe_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
          N = spa_num_patches+spe_num_patches+2
          self.cs = -1
        else:
          N = spa_num_patches+spe_num_patches
          self.cs = N

        self.blocks = nn.ModuleList([
                spectral_spatial_block(embed_dim, bi, N=N, drop_path = drop_path, counter=i, cls = self.cls, fu = fu) for i in range(depth)
                        ])
        self.head = nn.Linear(embed_dim, nclass)
        self.spa_pos_embed = nn.Parameter(torch.zeros(1, spa_num_patches+1, embed_dim), requires_grad=False)
        self.spe_pos_embed = nn.Parameter(positional_embedding_1d(spe_num_patches+1, embed_dim), requires_grad=False)

        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm
        self.initialize_weights()
        

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        spa_pos_embed = get_2d_sincos_pos_embed(self.spa_pos_embed.shape[-1], int(self.spa_patch_embed.num_patches**.5), cls_token=True)
        self.spa_pos_embed.data.copy_(torch.from_numpy(spa_pos_embed).float().unsqueeze(0))


    def forward_features(self, x):
        x_spa = self.dimen_redu(x)
        x_spa = self.spa_patch_embed(x_spa)
        x_spa = x_spa + self.spa_pos_embed[:, :-1, :]
    
        # append cls token
        if self.cls:
          spa_cls_token = self.spa_cls_token + self.spa_pos_embed[:, -1:, :]
          spa_cls_tokens = spa_cls_token.expand(x_spa.shape[0], -1, -1)
          x_spa = torch.cat((x_spa, spa_cls_tokens), dim=1)

        x_spe = self.spe_patch_embed(x[:,:,self.half_spa_patch_size-self.half_spe_patch_size:self.half_spa_patch_size+self.half_spe_patch_size+1,
                                       self.half_spa_patch_size-self.half_spe_patch_size:self.half_spa_patch_size+self.half_spe_patch_size+1])
        x_spe = x_spe + self.spe_pos_embed[:, :-1, :]
        # append cls token
        if self.cls:
          spe_cls_token = self.spe_cls_token + self.spe_pos_embed[:, -1:, :]
          spe_cls_tokens = spe_cls_token.expand(x_spe.shape[0], -1, -1)
          x_spe = torch.cat((x_spe, spe_cls_tokens), dim=1)

        for blk in self.blocks:
            x_spa, x_spe = blk(x_spa, x_spe)
        if self.global_pool:
            x_spa = x_spa[:, 0:self.cs, :].mean(dim=1)  # global pool without cls token
            x_spe = x_spe[:, 0:self.cs, :].mean(dim=1)
            outcome = self.fc_norm((x_spa + x_spe)/2)
        else:
            outcome = (x_spa[:, -1] + x_spe[:, -1])/2
        return outcome

    def forward(self, x):
        x = x.squeeze(1) 
        x = self.forward_features(x)
        out = self.head(x)  
            
        return out
    
