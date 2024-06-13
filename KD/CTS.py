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


factor = 2
init_embd_tt = init_embd_tc= 12
init_embd_s = init_embd_tc//2


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
        Dattention4,Dconv4 = self.final_up(Dconv3)
        out = self.output(Dconv4)    
        return x0,Econv1,Eattention1,Econv2,Eattention2,Econv3,Eattention3,Econv4,Eattention4,Dattention0,Dconv0,Dattention1,Dconv1,Dattention2,Dconv2,Dattention3,Dconv3,Dattention4,Dconv4,out

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
        print('xxx',x.shape)
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
        
    
        
        # self.Stats_F = module_1()
                
    def forward(self, inp):
        x0S,Econv1S,Eattention1S,Econv2S,Eattention2S,Econv3S,Eattention3S,Econv4S,Eattention4S,Dattention0S,Dconv0S,Dattention1S,Dconv1S,Dattention2S,Dconv2S,Dattention3S,Dconv3S,Dattention4S,Dconv4S,outS = self.student_model(inp)
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
        
        
        
        print('Student x0S = ',x0S.shape,'T_1 = ',x0_T1.shape,x0_C.shape)
        print('Student Econv1S = ',Econv1S.shape,'T_1 = ',x1_C.shape)
        print('Student Eattention1S  = ',Eattention1S.shape,'T_1 = ',Eattention1_T1.shape)
        print('Student Econv2S  = ',Econv2S.shape,'T_1 = ',x2_C.shape)
        print('Student Eattention2S =  ',Eattention2S.shape,'T_1 = ',Eattention2_T1.shape)
        print('Student Econv3S = ',Econv3S.shape,'T_1 = ',x3_C.shape)
        print('Student Eattention3S = ',Eattention3S.shape,'T_1 = ',Eattention3_T1.shape)
        print('Student Econv4S  = ',Econv4S.shape,'T_1 = ',x4_C.shape)
        print('Student Eattention4S = ',Eattention4S.shape,'T_1 = ',Eattention4_T1.shape)
        
        
        print('DECODER SIDE')
        
        
        print('Student Dattention0S = ',Dattention0S.shape,'T_1 = ',Dattention0_T1.shape)
        print('Student Dconv0S = ',Dconv0S.shape,'T_1 = ',y0_C.shape)
        print('Student Dattention1S  = ',Dattention1S.shape,'T_1 = ',Dattention1_T1.shape)
        print('Student Dconv1S  = ',Dconv1S.shape,'T_1 = ',y1_C.shape)
        print('Student Dattention2S =  ',Dattention2S.shape,'T_1 = ',Dattention2_T1.shape)
        print('Student Dconv2S = ',Dconv2S.shape,'T_1 = ',y2_C.shape)
        print('Student Dattention3S = ',Dattention3S.shape,'T_1 = ',Dattention3_T1.shape)
        print('Student Dconv3S  = ',Dconv3S.shape,'T_1 = ',y3_C.shape)
        print('Student Dattention4S = ',Dattention4S.shape,'T_1 = ',Dattention4_T1.shape)
        
        print('Student Dconv4S  = ',Dconv4S.shape,'T_1 = ',y4_C.shape)
        print('Student outS = ',outS.shape,'T_1 = ',out_T1.shape,out_C.shape)


        
   

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def model() -> Teacher_Student:
    model = Teacher_Student()
    model.to(device=DEVICE,dtype=torch.float)
    return model
from torchsummary import summary
model = model()
summary(model, [(1,160,160)])

 
    
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# def model() -> SNet:
#     model = SNet()
#     model.to(device=DEVICE,dtype=torch.float)
#     return model
# from torchsummary import summary
# model = model()
# summary(model, [(1,160,160)])

# def model() -> TTNet:
#     model = TTNet()
#     model.to(device=DEVICE,dtype=torch.float)
#     return model
# from torchsummary import summary
# model = model()
# summary(model, [(1,160,160)])

# def model() -> TCNet:
#     model = TCNet()
#     model.to(device=DEVICE,dtype=torch.float)
#     return model
# from torchsummary import summary
# model = model()
# summary(model, [(1,160,160)])
