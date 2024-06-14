factor = 4
init_embd_tt = init_embd_tc= 96
init_embd_s = init_embd_tc//factor


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from einops import rearrange
import math
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


img_size = 160
patch_size = 2
heads = 2

class PatchEmbed(nn.Module): # [2,1,160,160] -->[2,1600,96]
    def __init__(self, img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
            
        else:
            self.norm = None
            
        self.pos_embed =  positionalencoding1d(embed_dim,(img_size[0]//2)**2).to(DEVICE)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
               
        x = x + self.pos_embed 
        x = x.permute(0,2,1)
        x = x.view(B, self.embed_dim, H//patch_size, W//patch_size)
        return x
    
    
    
class Attention(nn.Module):

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        
        x = self.fc1(
                x
        ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, out_features)

        return x



class MLP1(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        
        x = self.fc1(
                x
        ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, out_features)

        return x
    
    

class Block(nn.Module):
    def __init__(self, dim,n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.,up=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.up = up
        self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )
        
        if self.up is not None:
            self.mlp_expand = MLP(
                    in_features=dim,
                    hidden_features=hidden_features,
                    out_features=dim//2,
            )

    def forward(self, x):
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) 

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        if self.up is not None:
            x = self.mlp_expand(x)
            x = x.permute(0,2,1)
            x = x.view(B, C//2 , H, W)
                
        if self.up is None:
            x = x.permute(0,2,1)
            x = x.view(B, C , H, W)

        return x

class PatchMerging(nn.Module):  # [2,1600,96] -->[2,400,192]
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):

        B, C, H, W = x.shape
        x = x.permute(0,2,3,1)


        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)        
        x = x.permute(0,2,1)
        x = x.view(B, self.dim*2 , H//2, W//2)
        
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim 
        self.expand = nn.Linear(dim, 2*dim, bias=False) 
        self.norm = norm_layer(dim//2)
        
    def forward(self, x):

        b,c,h,w = x.shape
        x = x.flatten(2).transpose(1, 2) 
        x = self.expand(x)
        B,L, C = x.shape
        x = x.view(B, h, w, C)
        
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x = self.norm(x)
        x = x.transpose(1, 2) 
        x = x.view(B, C//4,2*h, 2*w)
        return x

class PatchExpand_final2(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim 
        self.expand = nn.Linear(dim, 4*dim, bias=False) 
        self.norm = norm_layer(dim)
        
    def forward(self, x):

        b,c,h,w = x.shape
        x = x.flatten(2).transpose(1, 2) 
        x = self.expand(x)
        B,L, C = x.shape
        x = x.view(B, h, w, C)
        
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x = self.norm(x)
        x = x.transpose(1, 2) 
        x = x.view(B, C//4,2*h, 2*w)
        return x
    
    
class PatchExpand_final4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.flatten(2).transpose(1, 2) 
        
        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, h, w, C)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)
        x = x.transpose(1, 2)
        x = x.view(B, C//(self.dim_scale**2),4*h, 4*w)
        return x

class Down(nn.Module):
    def __init__(self, embed_dim,n_heads,mlp_ratio,qkv_bias,p,attn_p):
        super().__init__()
        self.c = DoubleConv(embed_dim, embed_dim)
        self.att = nn.Sequential(
            Block(dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p),
            Block(dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p),
        )
        self.pm = PatchMerging(embed_dim)
    def forward(self, x):
        conv = self.c(x)
        attention = self.att(conv)
        patch_merg = self.pm(attention)
        return conv,attention,patch_merg
    
class Down_t(nn.Module):
    def __init__(self, embed_dim,n_heads,mlp_ratio,qkv_bias,p,attn_p):
        super().__init__()
        self.att = nn.Sequential(
            Block(dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p),
            Block(dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p),
        )
        self.pm = PatchMerging(embed_dim)
    def forward(self, x):
        attention = self.att(x)
        patch_merg = self.pm(attention)
        return attention,patch_merg


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class StrideConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,stride =2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Up(nn.Module):
    def __init__(self, embed_dim,n_heads,mlp_ratio,qkv_bias,p,attn_p,up=None):
        super().__init__()
        
        self.up = PatchExpand(embed_dim)
        self.att =  nn.Sequential(
            Block(dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,up=None),
            Block(dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,up=up),
                                 )
        self.conv = DoubleConv(embed_dim, embed_dim//2)
    def forward(self, x2,x1):
        x2 = self.up(x2)
        c =  torch.cat((x1, x2),1)
        attention = self.att(c)
        conv = self.conv(attention)
        return attention,conv
    
class Up_t(nn.Module):
    def __init__(self, embed_dim,n_heads,mlp_ratio,qkv_bias,p,attn_p,up=None):
        super().__init__()
        
        self.up = PatchExpand(embed_dim)
        self.att =  nn.Sequential(
            Block(dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,up=None),
            Block(dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,up=up),
                                 )
    def forward(self, x2,x1):
        x2 = self.up(x2)
        c =  torch.cat((x1, x2),1)
        attention = self.att(c)
        return attention
    
class Down_c(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.d_conv = DoubleConv(in_channels, in_channels)
        self.st_conv = StrideConv(in_channels,out_channels)

    def forward(self, x):
        c = self.d_conv(x)
        x = self.st_conv(c)
        return c,x

class Up_c(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
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
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()
        
        self.patch_embd  = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=1,
                embed_dim=init_embd_s,
        )
        self.down1 = Down(embed_dim=init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        
        self.down2 = Down(embed_dim=2*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.down3 = Down(embed_dim=4*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.down4 = Down(embed_dim=8*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        
        self.up1 = Up(embed_dim=16*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.up2 = Up(embed_dim=8*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.up3 = Up(embed_dim=4*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.up4 = Up(embed_dim=2*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        
        self.final_up = PatchExpand_final2(init_embd_s)
        self.output = nn.Conv2d(init_embd_s, 5, kernel_size=1)
                
    def forward(self, inp):
        x0 = self.patch_embd(inp) 
        Econv1,Eattention1,x1 = self.down1(x0)
        Econv2,Eattention2,x2 = self.down2(x1)
        Econv3,Eattention3,x3 = self.down3(x2)
        Econv4,Eattention4,x4 = self.down4(x3)
        
        Dattention0,Dconv0 = self.up1(x4,x3)
        Dattention1,Dconv1 = self.up2(Dconv0,x2)        
        Dattention2,Dconv2 = self.up3(Dconv1,x1)        
        Dattention3,Dconv3 = self.up4(Dconv2,x0)        
        Dattention4_f = self.final_up(Dconv3)
        out = self.output(Dattention4_f)    
        return x0,Econv1,Eattention1,Econv2,Eattention2,Econv3,Eattention3,Econv4,Eattention4,Dattention0,Dconv0,Dattention1,Dconv1,Dattention2,Dconv2,Dattention3,Dconv3,Dattention4_f,out

class TTNet(nn.Module):
    def __init__(self):
        super(TTNet, self).__init__()
        
        self.patch_embd  = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=1,
                embed_dim=init_embd_tt,
        )
        self.down1 = Down_t(embed_dim=init_embd_tt,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        
        self.down2 = Down_t(embed_dim=2*init_embd_tt,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.down3 = Down_t(embed_dim=4*init_embd_tt,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.down4 = Down_t(embed_dim=8*init_embd_tt,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        
        self.up1 = Up_t(embed_dim=16*init_embd_tt,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.,up=True)
        self.up2 = Up_t(embed_dim=8*init_embd_tt,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.,up=True)
        self.up3 = Up_t(embed_dim=4*init_embd_tt,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.,up=True)
        self.up4 = Up_t(embed_dim=2*init_embd_tt,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.,up=True)
        
        self.final_up = PatchExpand_final2(init_embd_tt)
        self.output = nn.Conv2d(init_embd_tt, 5, kernel_size=1)
                
    def forward(self, inp):
        x0 = self.patch_embd(inp)
        Eattention1,x1 = self.down1(x0)
        Eattention2,x2 = self.down2(x1)
        Eattention3,x3 = self.down3(x2)
        Eattention4,x4 = self.down4(x3)
        
        Dattention0 = self.up1(x4,x3)
        Dattention1 = self.up2(Dattention0,x2)        
        Dattention2 = self.up3(Dattention1,x1)        
        Dattention3 = self.up4(Dattention2,x0)        
        Dattention4 = self.final_up(Dattention3)
        out = self.output(Dattention4)    
        return x0,Eattention1,Eattention2,Eattention3,Eattention4,Dattention0,Dattention1,Dattention2,Dattention3,Dattention4,out

class TCNet(nn.Module):
    def __init__(self, n_channels=1, bilinear=False):
        super(TCNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        
        
        self.patch_embd  = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=1,
                embed_dim=init_embd_tt,
        )

        self.down1 = Down_c(n_channels, init_embd_tc)
        self.down2 = Down_c(init_embd_tc, 2*init_embd_tc)
        self.down3 = Down_c(2*init_embd_tc, 4*init_embd_tc)
        factor = 2 if bilinear else 1
        self.down4 = Down_c(4*init_embd_tc, 8*init_embd_tc // factor)
        self.down5 = Down_c(8*init_embd_tc, 16*init_embd_tc // factor)
        
        self.up1 = Up_c(16*init_embd_tc, 8*init_embd_tc // factor, bilinear)
        self.up2 = Up_c(8*init_embd_tc, 4*init_embd_tc // factor, bilinear)
        self.up3 = Up_c(4*init_embd_tc, 2*init_embd_tc // factor, bilinear)
        self.up4 = Up_c(2*init_embd_tc, init_embd_tc, bilinear)
        self.up5 = nn.ConvTranspose2d(init_embd_tc, init_embd_tc, kernel_size=2, stride=2)
        self.outc = nn.Conv2d(init_embd_tc, 5, kernel_size=1)
        
    def forward(self, inp):
        
        x0 = self.patch_embd(inp)
        c1,x1 = self.down2(x0)
        c2,x2 = self.down3(x1)
        c3,x3 = self.down4(x2)
        c4,x4 = self.down5(x3)
        
        y1 = self.up1(x4, x3)
        y2 = self.up2(y1, x2)
        y3 = self.up3(y2, x1)
        y4 = self.up4(y3, x0)
        y5 = self.up5(y4)
        out = self.outc(y5)
        return x0,c1,c2,c3,c4,y1,y2,y3,y4,y5,out


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Mlp_E(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mlp_E, self).__init__()
        self.mlp = MLP1(
                in_features=in_channels,
                hidden_features=out_channels//2,
                out_features=out_channels,
        )
        
        self.out_channels = out_channels
        
    def forward(self, x):
        
        b,c,h,w = x.shape
        x = x.flatten(2).transpose(1, 2) 
        x = self.mlp(x)
        x = x.permute(0,2,1)
        x = x.view(b, self.out_channels , h, w)
        return x
    
    
class Teacher_Student(nn.Module):
    def __init__(self, n_channels=1):
        super(Teacher_Student, self).__init__()
        self.n_channels = n_channels
        
        self.student_model = SNet()
        self.teacherT_model = TTNet()
        self.teacherC_model = TCNet()
        
        
        
        self.c1 = Conv(init_embd_tt,init_embd_tt//factor)
        self.c2 = Conv(init_embd_tt,init_embd_tt//factor)
        self.c3 = Conv(2*init_embd_tt,2*init_embd_tt//factor)
        self.c4 = Conv(4*init_embd_tt,4*init_embd_tt//factor)
        self.c5 = Conv(8*init_embd_tt,8*init_embd_tt//factor)
        
        
        self.d1 = Conv(init_embd_tt,init_embd_tt//factor)
        self.d2 = Conv(init_embd_tt,init_embd_tt//factor)
        self.d3 = Conv(2*init_embd_tt,2*init_embd_tt//factor)
        self.d4 = Conv(4*init_embd_tt,4*init_embd_tt//factor)
        self.d5 = Conv(8*init_embd_tt,8*init_embd_tt//factor)
                
        self.a1 = Mlp_E(init_embd_tt,init_embd_tt//factor)
        self.a2 = Mlp_E(init_embd_tt,init_embd_tt//factor)
        self.a3 = Mlp_E(2*init_embd_tt,2*init_embd_tt//factor)
        self.a4 = Mlp_E(4*init_embd_tt,4*init_embd_tt//factor)
        self.a5 = Mlp_E(8*init_embd_tt,8*init_embd_tt//factor)
        
        self.dAt4 = Mlp_E(init_embd_tt,init_embd_tt//factor)
        self.dAt3 = Mlp_E(init_embd_tt,2*init_embd_tt//factor)
        self.dAt2 = Mlp_E(2*init_embd_tt,4*init_embd_tt//factor)
        self.dAt1 = Mlp_E(4*init_embd_tt,8*init_embd_tt//factor)
        
        # self.Stats_F = module_1()
                
    def forward(self, inp):
        x0S,Econv1S,Eattention1S,Econv2S,Eattention2S,Econv3S,Eattention3S,Econv4S,Eattention4S,Dattention0S,Dconv0S,Dattention1S,Dconv1S,Dattention2S,Dconv2S,Dattention3S,Dconv3S,Dattention4S_final,outS = self.student_model(inp)
        x0_T1,Eattention1_T1,Eattention2_T1,Eattention3_T1,Eattention4_T1,Dattention0_T1,Dattention1_T1,Dattention2_T1,Dattention3_T1,Dattention4_T1,out_T1 = self.teacherT_model(inp)
        x0_C,x1_C,x2_C,x3_C,x4_C,y0_C,y1_C,y2_C,y3_C,y4_C,out_C = self.teacherC_model(inp)
        
        
        x0_C = self.c1(x0_C)
        x1_C = self.c2(x1_C)
        x2_C = self.c3(x2_C)
        x3_C = self.c4(x3_C)
        x4_C = self.c5(x4_C)
        
        x0_T1 = self.a1(x0_T1)
        Eattention1_T1 = self.a2(Eattention1_T1)
        Eattention2_T1 = self.a3(Eattention2_T1)
        Eattention3_T1 = self.a4(Eattention3_T1)
        Eattention4_T1 = self.a5(Eattention4_T1)
        
        
        y0_C = self.d5(y0_C)
        y1_C = self.d4(y1_C)
        y2_C = self.d3(y2_C)
        y3_C = self.d2(y3_C)
        y4_C = self.d1(y4_C)
        
        Dattention4_T1 = self.dAt4(Dattention4_T1)
        Dattention3_T1 = self.dAt3(Dattention3_T1)
        Dattention2_T1 = self.dAt2(Dattention2_T1)
        
        student = [x0S,Econv1S,Eattention1S,Econv2S,Eattention2S,Econv3S,Eattention3S,Econv4S,Eattention4S,Dattention0S,Dconv0S,Dattention1S,Dconv1S,Dattention2S,Dconv2S,Dattention3S,Dconv3S,Dattention4S_final,outS]
        teacher_T = [x0_T1,Eattention1_T1,Eattention2_T1,Eattention3_T1,Eattention4_T1,Dattention0_T1,Dattention1_T1,Dattention2_T1,Dattention3_T1,Dattention4_T1,out_T1]
        teacher_C = [x0_C,x1_C,x2_C,x3_C,x4_C,y0_C,y1_C,y2_C,y3_C,y4_C,out_C]
        

        # print('Student x0S = ',x0S.shape,'T_1 = ',x0_T1.shape,x0_C.shape)
        # print('Student Econv1S = ',Econv1S.shape,'T_1 = ',x1_C.shape)
        # print('Student Eattention1S  = ',Eattention1S.shape,'T_1 = ',Eattention1_T1.shape)
        # print('Student Econv2S  = ',Econv2S.shape,'T_1 = ',x2_C.shape)
        # print('Student Eattention2S =  ',Eattention2S.shape,'T_1 = ',Eattention2_T1.shape)
        # print('Student Econv3S = ',Econv3S.shape,'T_1 = ',x3_C.shape)
        # print('Student Eattention3S = ',Eattention3S.shape,'T_1 = ',Eattention3_T1.shape)
        # print('Student Econv4S  = ',Econv4S.shape,'T_1 = ',x4_C.shape)
        # print('Student Eattention4S = ',Eattention4S.shape,'T_1 = ',Eattention4_T1.shape)
        print('DECODER SIDE')
        print('Student Dattention0S = ',Dattention0S.shape,'T_1 = ',Dattention0_T1.shape)
        print('Student Dconv0S = ',Dconv0S.shape,'T_1 = ',y0_C.shape)
        print('Student Dattention1S  = ',Dattention1S.shape,'T_1 = ',Dattention1_T1.shape)
        print('Student Dconv1S  = ',Dconv1S.shape,'T_1 = ',y1_C.shape)
        print('Student Dattention2S =  ',Dattention2S.shape,'T_1 = ',Dattention2_T1.shape)
        print('Student Dconv2S = ',Dconv2S.shape,'T_1 = ',y2_C.shape)
        # print('Student Dattention3S = ',Dattention3S.shape,'T_1 = ',Dattention3_T1.shape)
        # print('Student Dconv3S  = ',Dconv3S.shape,'T_1 = ',y3_C.shape)
        # print('Student Dattention4S = ',Dattention4S_final.shape,'T_1 = ',Dattention4_T1.shape,'  ',y4_C.shape)
        # print('Student outS = ',outS.shape,'T_1 = ',out_T1.shape,out_C.shape)
        
        return  student,teacher_T,teacher_C
        

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def model() -> Teacher_Student:
    model = Teacher_Student()
    model.to(device=DEVICE,dtype=torch.float)
    return model
from torchsummary import summary
model = model()
summary(model, [(1,160,160)])



'''
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import SimpleITK as sitk
import os
import torch
import matplotlib.pyplot as plt
#from typing import List, Union, Tuple

import torchio as tio
           ###########  Dataloader  #############
NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 160
   
  
    
def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_):# org_dim3->numof channels
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_


def Normalization_1(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 

geometrical_transforms = tio.OneOf([
    tio.RandomFlip(axes=([1, 2])),
    #tio.RandomElasticDeformation(num_control_points=(5, 5, 5), locked_borders=1, image_interpolation='nearest'),
    tio.RandomAffine(degrees=(-45, 45), center='image'),
])

intensity_transforms = tio.OneOf([
    tio.RandomBlur(),
    tio.RandomGamma(log_gamma=(-0.2, -0.2)),
    tio.RandomNoise(mean=0.1, std=0.1),
    tio.RandomGhosting(axes=([1, 2])),
])

transforms_2d = tio.Compose({
    geometrical_transforms: 0.3,  # Probability for geometric transforms
    intensity_transforms: 0.3,   # Probability for intensity transforms
    tio.Lambda(lambda x: x): 0.4 # Probability for no augmentation (original image)
})
   
def generate_label(gt):
        temp_ = np.zeros([5,DIM_,DIM_])
        temp_[0:1,:,:][np.where(gt==1)]=1
        temp_[1:2,:,:][np.where(gt==2)]=1
        temp_[2:3,:,:][np.where(gt==3)]=1
        temp_[3:4,:,:][np.where(gt==4)]=1
        temp_[4:5,:,:][np.where(gt==0)]=1
        return temp_


class Dataset_io(Dataset): 
    def __init__(self, images_folder,transformations=transforms_2d):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        self.images_name = os.listdir(images_folder)
        self.transformations = transformations
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3)) 
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_1(img)
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-11]+'_gt.nii.gz'        
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        gt = gt.astype(np.float64)
        
        gt = np.expand_dims(gt, axis=0)
        img = np.expand_dims(img, axis=0)
        
        C = img.shape[0]
        H = img.shape[1]
        W = img.shape[2]
        img = Cropping_3d(C,H,W,DIM_,img)
        
        C = gt.shape[0]
        H = gt.shape[1]
        W = gt.shape[2]
        gt = Cropping_3d(C,H,W,DIM_,gt)
        
        ## apply augmentaitons here ###
        
        img = np.expand_dims(img, axis=3)
        gt = np.expand_dims(gt, axis=3)

        d = {}
        d['Image'] = tio.Image(tensor = img, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = gt, type=tio.LABEL)
        sample = tio.Subject(d)
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img = transformed_tensor['Image'].data
            gt = transformed_tensor['Mask'].data
    
        gt = gt[...,0]
        img = img[...,0] 
        
        gt = generate_label(gt)

        return img,gt
    
def Data_Loader_io_transforms(images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

    
class Dataset_val(Dataset): 
    def __init__(self, images_folder):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        self.images_name = os.listdir(images_folder)
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3)) 
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_1(img)
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-11]+'_gt.nii.gz'        
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        gt = gt.astype(np.float64)
        
        gt = np.expand_dims(gt, axis=0)
        img = np.expand_dims(img, axis=0)
        
        C = img.shape[0]
        H = img.shape[1]
        W = img.shape[2]
        img = Cropping_3d(C,H,W,DIM_,img)
        
        C = gt.shape[0]
        H = gt.shape[1]
        W = gt.shape[2]
        gt = Cropping_3d(C,H,W,DIM_,gt)
        gt = generate_label(gt)

        return img,gt
        
def Data_Loader_val(images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val(images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

# val_imgs = r'C:\My_Data\TMI\Second_Data\Only_4Chamber\F1\val\imgs/' ## path to images
# train_loader = Data_Loader_io_transforms(val_imgs,batch_size = 1)
# a = iter(train_loader)
# for i in range(21):
#     a1 =next(a) 
#     img = a1[0].numpy()
#     gt = a1[1].numpy()
#     plt.figure()
#     plt.imshow(img[0,0,:])
#     for k in range(5):
#         plt.figure()
#         plt.imshow(gt[0,k,:])


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
import torch.optim as optim

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def cal_dice(pre_2d,gt):
    pred = torch.argmax(pre_2d, dim=1)

    out_LA = torch.zeros_like(pred)
    out_LA[torch.where(pred==0)] = 1
                
    out_RA = torch.zeros_like(pred)
    out_RA[torch.where(pred==1)] = 1
                
    out_LV = torch.zeros_like(pred)
    out_LV[torch.where(pred==2)] = 1
            
    out_RV = torch.zeros_like(pred)
    out_RV[torch.where(pred==3)] = 1
            
                        
    single_LA = (2 * (out_LA * gt[:,0,:]).sum()) / (
               (out_LA + gt[:,0,:]).sum() + 1e-8)
            
            
    single_RA = (2 * (out_RA * gt[:,1,:]).sum()) / (
               (out_RA + gt[:,1,:]).sum() + 1e-8)
            
    single_LV = (2 * (out_LV * gt[:,2,:]).sum()) / (
               (out_LV + gt[:,2,:]).sum() + 1e-8)
            
    single_RV = (2 * (out_RV * gt[:,3,:]).sum()) / (
               (out_RV + gt[:,3,:]).sum() + 1e-8)
    
    return single_LA,single_RA,single_LV,single_RV
            
            
def check_Dice_Score(loader, model1, device=DEVICE):
    
    Dice_score_LA_S = 0
    Dice_score_RA_S = 0
    Dice_score_LV_S = 0
    Dice_score_RV_S = 0
    
    
    Dice_score_LA_TT = 0
    Dice_score_RA_TT = 0
    Dice_score_LV_TT = 0
    Dice_score_RV_TT = 0
    
    Dice_score_LA_TC = 0
    Dice_score_RA_TC = 0
    Dice_score_LV_TC = 0
    Dice_score_RV_TC = 0
    
        
    loop = tqdm(loader)
    model1.eval()
    
    for batch_idx, (img,gt) in enumerate(loop):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
        
        with torch.no_grad(): 
            
            s,t_t,t_c  = model1(img)

            single_LA,single_RA,single_LV,single_RV = cal_dice(s[18],gt)
            Dice_score_LA_S +=single_LA
            Dice_score_RA_S +=single_RA
            Dice_score_LV_S +=single_LV
            Dice_score_RV_S +=single_RV
            
            single_LA,single_RA,single_LV,single_RV = cal_dice(t_t[10],gt)
            Dice_score_LA_TT +=single_LA
            Dice_score_RA_TT +=single_RA
            Dice_score_LV_TT+=single_LV
            Dice_score_RV_TT +=single_RV
            
            single_LA,single_RA,single_LV,single_RV = cal_dice(t_c[10],gt)
            Dice_score_LA_TC +=single_LA
            Dice_score_RA_TC +=single_RA
            Dice_score_LV_TC+=single_LV
            Dice_score_RV_TC +=single_RV
            
    ## segemntaiton ##
    print(f"Dice_score_LA_S  : {Dice_score_LA_S/len(loader)}")
    print(f"Dice_score_RA_S  : {Dice_score_RA_S/len(loader)}")
    print(f"Dice_score_LV_S  : {Dice_score_LV_S/len(loader)}")
    print(f"Dice_score_RV_S  : {Dice_score_RV_S/len(loader)}")
    Overall_Dicescore__S = (Dice_score_LA_S + Dice_score_RA_S + Dice_score_LV_S + Dice_score_RV_S )/4
    print(f"Overall_Dicescore__S  : {Overall_Dicescore__S/len(loader)}")
    
    
    print(f"Dice_score_LA_TT  : {Dice_score_LA_TT/len(loader)}")
    print(f"Dice_score_RA_TT  : {Dice_score_RA_TT/len(loader)}")
    print(f"Dice_score_LV_TT  : {Dice_score_LV_TT/len(loader)}")
    print(f"Dice_score_RV_TT  : {Dice_score_RV_TT/len(loader)}")
    Overall_Dicescore__TT = (Dice_score_LA_TT + Dice_score_RA_TT + Dice_score_LV_TT + Dice_score_RV_TT)/4
    print(f"Overall_Dicescore__TT  : {Overall_Dicescore__TT/len(loader)}")
    
    
    print(f"Dice_score_LA_TC  : {Dice_score_LA_TC/len(loader)}")
    print(f"Dice_score_RA_TC  : {Dice_score_RA_TC/len(loader)}")
    print(f"Dice_score_LV_TC  : {Dice_score_LV_TC/len(loader)}")
    print(f"Dice_score_RV_TC  : {Dice_score_RV_TC/len(loader)}")
    Overall_Dicescore__TC = (Dice_score_LA_TC+ Dice_score_RA_TC + Dice_score_LV_TC + Dice_score_RV_TC )/4
    print(f"Overall_Dicescore__TC  : {Overall_Dicescore__TC/len(loader)}")
    
    return Overall_Dicescore__S/len(loader)




import torch.nn as nn    
mse_loss = nn.MSELoss()

def LossC_Module(s1,s2,s3,s4,s5,s6,s7,s8,s9,t1,t2,t3,t4,t5,t6,t7,t8,t9):
    
    E1 = mse_loss(s1,t1)
    E2 = mse_loss(s2,t2)
    E3 = mse_loss(s3,t3)
    E4 = mse_loss(s4,t4)
    E5 = mse_loss(s5,t5)
    
    D1 = mse_loss(s6,t6)
    D2 = mse_loss(s7,t7)
    D3 = mse_loss(s8,t8)
    D4 = mse_loss(s9,t9)
    
    Avg = (E1+E2+E3+E4+E5+D1+D2+D3+D4)/9
    return Avg


def LossT_Module(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10):
    
    E1 = mse_loss(s1,t1)
    E2 = mse_loss(s2,t2)
    E3 = mse_loss(s3,t3)
    E4 = mse_loss(s4,t4)
    E5 = mse_loss(s5,t5)
    
    D1 = mse_loss(s6,t6)
    D2 = mse_loss(s7,t7)
    D3 = mse_loss(s8,t8)
    D4 = mse_loss(s9,t9)
    D6 = mse_loss(s10,t10)
    
    Avg = (E1+E2+E3+E4+E5+D1+D2+D3+D4+D6)/10
    return Avg

def train_fn(loader_train1,loader_valid1,model1, optimizer1, scaler1,loss_fn_DC1,scheduler): ### Loader_1--> ED and Loader2-->ES

    train_losses1_seg  = [] # loss of each batch
    valid_losses1_seg  = []  # loss of each batch
    
    
    loop = tqdm(loader_train1)
    model1.train()
    
    
    for batch_idx,(img,gt)  in enumerate(loop):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
       

        with torch.cuda.amp.autocast():
            s,t_t,t_c  = model1(img)
            
            out_S = loss_fn_DC1(s[18],gt)
            out_TT = loss_fn_DC1(t_t[10],gt)
            out_TC = loss_fn_DC1(t_c[10],gt)
            
            seg_loss = (out_S + out_TT + out_TC)/3
            
            Loss_C = LossC_Module(s[0],s[1],s[3],s[5],s[7],s[10],s[12],s[14],s[16],t_c[0],t_c[1],t_c[2],t_c[3],t_c[4],t_c[5],t_c[6],t_c[7],t_c[8])
            Loss_T = LossT_Module(s[0],s[2],s[4],s[6],s[8],s[9],s[11],s[13],s[15],s[17],t_t[0],t_t[1],t_t[2],t_t[3],t_t[4],t_t[5],t_t[6],t_t[7],t_t[8],t_t[9])
            
            mse_loss = 0.5*Loss_C + 0.5*Loss_T
            
            loss = 0.5*seg_loss + 0.5*mse_loss
            
        optimizer1.zero_grad()        
        scaler1.scale(loss).backward()        
        scaler1.step(optimizer1)        
        scaler1.update()
        
        
        # update tqdm loop
        loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
        
        train_losses1_seg.append(float(loss))
        
    loop_v = tqdm(loader_valid1)
    model1.eval() 
    
    for batch_idx,(img,gt) in enumerate(loop_v):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
       
        with torch.no_grad(): 
            s,t_t,t_c  = model1(img)    
            
            out_S = loss_fn_DC1(s[18],gt)
            out_TT = loss_fn_DC1(t_t[10],gt)
            out_TC = loss_fn_DC1(t_c[10],gt)
            
            seg_loss = (out_S + out_TT + out_TC)/3
            
            Loss_C = LossC_Module(s[0],s[1],s[3],s[5],s[7],s[10],s[12],s[14],s[16],t_c[0],t_c[1],t_c[2],t_c[3],t_c[4],t_c[5],t_c[6],t_c[7],t_c[8])
            Loss_T = LossT_Module(s[0],s[2],s[4],s[6],s[8],s[9],s[11],s[13],s[15],s[17],t_t[0],t_t[1],t_t[2],t_t[3],t_t[4],t_t[5],t_t[6],t_t[7],t_t[8],t_t[9])
            
            mse_loss = 0.5*Loss_C + 0.4*Loss_T
            
            loss = 0.5*seg_loss + 0.5*mse_loss
            
        # backward
        loop_v.set_postfix(loss = loss.item())
        valid_losses1_seg.append(float(loss))

    train_loss_per_epoch1_seg = np.average(train_losses1_seg)
    valid_loss_per_epoch1_seg  = np.average(valid_losses1_seg)
    
    avg_train_losses1_seg.append(train_loss_per_epoch1_seg)
    avg_valid_losses1_seg.append(valid_loss_per_epoch1_seg)
    
    #print(get_lr(optimizer1))
    
    
    return train_loss_per_epoch1_seg,valid_loss_per_epoch1_seg

  
from DC1 import DiceLoss
loss_fn_DC1 = DiceLoss()
#from m1 import SSM_UNET


for fold in range(1,6):

  #from m1 import Teacher_Student
  model_1 =  Teacher_Student()


  fold = str(fold)  ## training fold number 
  
  train_imgs = "/data/scratch/acw676/ST_June/data1/five_folds/F"+fold+"/train/imgs/"
  val_imgs  = "/data/scratch/acw676/ST_June/data1/five_folds/F"+fold+"/val/imgs/"
  
  Batch_Size = 4
  Max_Epochs = 500
  
  train_loader = Data_Loader_io_transforms(train_imgs,batch_size = Batch_Size)
  val_loader = Data_Loader_val(val_imgs,batch_size = 1)
  
  
  print(len(train_loader)) ### this shoud be = Total_images/ batch size
  print(len(val_loader))   ### same here
  #print(len(test_loader))   ### same here
  
  avg_train_losses1_seg = []   # losses of all training epochs
  avg_valid_losses1_seg = []  #losses of all training epochs
    
  avg_valid_DS_ValSet_seg = []  # all training epochs
  avg_valid_DS_TrainSet_seg = []  # all training epochs
    
  path_to_save_check_points = '/data/scratch/acw676/ST_June/data1/W_ST/'+'/F'+fold+'_T'+str(init_embd_tt)+'_S'+str(init_embd_s)+'_F'+str(factor)
  path_to_save_Learning_Curve = '/data/scratch/acw676/ST_June/data1/W_ST/'+'/F'+fold+'_T'+str(init_embd_tt)+'_S'+str(init_embd_s)+'_F'+str(factor)

  
  ### 3 - this function will save the check-points 
  def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
      print("=> Saving checkpoint")
      torch.save(state, filename)
      
  epoch_len = len(str(Max_Epochs))
  
    # Variable to keep track of maximum Dice validation score

  def main():
      max_dice_val = 0.0
      model1 = model_1.to(device=DEVICE,dtype=torch.float)
      scaler1 = torch.cuda.amp.GradScaler()
      optimizer1 = optim.AdamW(model1.parameters(),betas=(0.5, 0.5),lr=0.0001) #  0.00005
      #optimizer1 = optim.AdamW(model1.parameters(),betas=(0.3, 0.33),lr=0.00005) #  0.00005
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer1,milestones=[50,100,200,400,500], gamma=0.5)#  milestones=[20,40,60,80,100,120,500], gamma=0.5)
      for epoch in range(Max_Epochs):
          
          train_loss_seg ,valid_loss_seg = train_fn(train_loader,val_loader, model1, optimizer1,scaler1,loss_fn_DC1,scheduler)
          scheduler.step()
          
          print_msg1 = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss_seg: {train_loss_seg:.5f} ' +
                     f'valid_loss_seg: {valid_loss_seg:.5f}')
        
          
          print(print_msg1)

          Dice_val = check_Dice_Score(val_loader, model1, device=DEVICE)
          avg_valid_DS_ValSet_seg.append(Dice_val.detach().cpu().numpy()) 
          
          
          if Dice_val > max_dice_val:
              max_dice_val = Dice_val
            # Save the checkpoint
              checkpoint = {
                  "state_dict": model1.state_dict(),
                  "optimizer": optimizer1.state_dict(),
                  }
              save_checkpoint(checkpoint)
          
  
  if __name__ == "__main__":
      main()
  
  fig = plt.figure(figsize=(10,8))
    
  plt.plot(range(1,len(avg_train_losses1_seg)+1),avg_train_losses1_seg, label='Training Segmentation Loss')
  plt.plot(range(1,len(avg_valid_losses1_seg)+1),avg_valid_losses1_seg,label='Validation Segmentation Loss')
  
  plt.plot(range(1,len(avg_valid_DS_ValSet_seg)+1),avg_valid_DS_ValSet_seg,label='Validation DS')

    # find position of lowest validation loss
  minposs = avg_valid_losses1_seg.index(min(avg_valid_losses1_seg))+1 
  plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')
  font1 = {'size':20}
  plt.title("Learning Curve Graph",fontdict = font1)
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.ylim(-1, 1) # consistent scale
  plt.xlim(0, len(avg_train_losses1_seg)+1) # consistent scale
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')'''
