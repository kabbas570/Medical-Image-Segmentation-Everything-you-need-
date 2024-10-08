factor = 4
init_embd_t = init_embd_tc = 24
init_embd_s = init_embd_tc//factor 


img_size = 256
patch_size = 4
heads = 2


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from einops import rearrange
import math
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
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
    
    
class PatchExpand(nn.Module): ## halves the channels and doubles the spatial dims
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
        
        x = self.fc1(x) 
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, out_features)
        x = self.act(x)
        x = self.drop(x)
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
    def __init__(self, dim,n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.,up=None,out_f=None):
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
                    out_features = out_f,
            )
            
            self.out_f = out_f

    def forward(self, x):
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) 

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        if self.up is not None:
            x = self.mlp_expand(x)
            x = x.permute(0,2,1)
            x = x.view(B, self.out_f  , H, W)
                
        if self.up is None:
            x = x.permute(0,2,1)
            x = x.view(B, C , H, W)

        return x    
class Down(nn.Module):
    def __init__(self, embed_dim,n_heads,mlp_ratio,qkv_bias,p,attn_p):
        super().__init__()
        self.conv = DoubleConv(embed_dim, embed_dim)
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
        convf = self.conv(x)
        attentionf = self.att(x)

        mergedf = convf + attentionf ## here we can do more complex ways of combining.
        
        patch_mergf = self.pm(mergedf)
        return convf,attentionf,patch_mergf

class Down_F(nn.Module):
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
        attentionf = self.att(x)
        attentionf = self.pm(attentionf)
        return attentionf
    
class Down_C(nn.Module):
    def __init__(self, embed_dim,n_heads,mlp_ratio,qkv_bias,p,attn_p):
        super().__init__()
        self.conv = DoubleConv(embed_dim, embed_dim)
        self.pm = PatchMerging(embed_dim)
    def forward(self, x):
        convf = self.conv(x)
        convf = self.pm(convf)
        return convf
    
class Down_S(nn.Module):
    def __init__(self, embed_dim,n_heads,mlp_ratio,qkv_bias,p,attn_p):
        super().__init__()
        self.conv = SingleConv(embed_dim, embed_dim)
        self.att = nn.Sequential(
            Block(dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p),
        )
        self.pm = PatchMerging(embed_dim)
    def forward(self, x):
        convf = self.conv(x)
        attentionf = self.att(x)
        
        mergedf = convf + attentionf ## here we can do more complex ways of combining.
        
        patch_mergf = self.pm(mergedf)
        return convf,attentionf,patch_mergf
    
# inp = torch.randn(4,16,64,64) 
# dow = Down(embed_dim=16,n_heads=2,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
# out = dow(inp)   
    
   
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
                    attn_p=attn_p,up='yes',out_f = embed_dim//2),
                                 )
        self.conv = DoubleConv(embed_dim, embed_dim//2)
    def forward(self, x2,x1):

        x2 = self.up(x2)
        c =  torch.cat((x1, x2),1)
        #c = self.conv_1(c)
        attentionf = self.att(c)
        convf = self.conv(c)
        mergedf = convf + attentionf ## here we can do more complex ways of combining.
        return convf,attentionf,mergedf

class Up_S(nn.Module):
    def __init__(self, embed_dim,n_heads,mlp_ratio,qkv_bias,p,attn_p,up=None):
        super().__init__()
        self.up = PatchExpand(embed_dim)
        self.att =  nn.Sequential(
            Block(dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,up='yes',out_f = embed_dim//2),
                                 )
        self.conv = SingleConv(embed_dim, embed_dim//2)
    def forward(self, x2,x1):
        x2 = self.up(x2)
        c =  torch.cat((x1, x2),1)
        attentionf = self.att(c)
        convf = self.conv(c)
        mergedf = convf + attentionf ## here we can do more complex ways of combining.
        return convf,attentionf,mergedf  

class Up_C(nn.Module):
    def __init__(self, embed_dim,n_heads,mlp_ratio,qkv_bias,p,attn_p,up=None):
        super().__init__()
        self.up = PatchExpand(embed_dim)
        self.conv = DoubleConv(embed_dim, embed_dim//2)
    def forward(self, x2,x1):
        x2 = self.up(x2)
        c =  torch.cat((x1, x2),1)
        convf = self.conv(c)
        return convf

class Up_F(nn.Module):
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
                    attn_p=attn_p,up='yes',out_f = embed_dim//2),
                                 )
    def forward(self, x2,x1):
        x2 = self.up(x2)
        c =  torch.cat((x1, x2),1)
        attentionf = self.att(c)
        return attentionf
    
    
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
            
        #self.pos_embed =  positionalencoding1d(embed_dim,(img_size[0]//2)**2).to(DEVICE)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
               
       # x = x + self.pos_embed 
        x = x.permute(0,2,1)
        x = x.view(B, self.embed_dim, H//patch_size, W//patch_size)
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

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        b,ch,h,w = x.shape
        x = x.flatten(2).transpose(1, 2) 

        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x = self.norm(x)
        x = x.transpose(1, 2) 
        
        x = x.view(B,ch,4*H, 4*W)
        
        return x
    
class Hybrid_Sudent(nn.Module):
    def __init__(self):
        super(Hybrid_Sudent, self).__init__()
        
        self.patch_embd  = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=1,
                embed_dim=init_embd_s,
        )
        self.down1 = Down_C(embed_dim=init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.down2 = Down_C(embed_dim=2*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.down3 = Down_C(embed_dim=4*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.up2 = Up_C(embed_dim=8*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.up3 = Up_C(embed_dim=4*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.up4 = Up_C(embed_dim=2*init_embd_s,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        
        self.final_up = FinalPatchExpand_X4([64,64],init_embd_s)
        self.output = nn.Conv2d(init_embd_s, 4, kernel_size=1)
                
    def forward(self, inp):
        x0 = self.patch_embd(inp) 
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
                
        f1 = self.up2(x3,x2)        
        f2 = self.up3(f1,x1)        
        f3 = self.up4(f2,x0)    
  
        f4 = self.final_up(f3)
        out = self.output(f4) 
        
        return x1,x2,x3,f1,f2,f3,out
    

class Hybrid_Teacher(nn.Module):
    def __init__(self):
        super(Hybrid_Teacher, self).__init__()
        
        self.patch_embd  = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=1,
                embed_dim=init_embd_t,
        )
        self.down1 = Down_C(embed_dim=init_embd_t,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.down2 = Down_C(embed_dim=2*init_embd_t,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.down3 = Down_C(embed_dim=4*init_embd_t,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.up2 = Up_C(embed_dim=8*init_embd_t,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.up3 = Up_C(embed_dim=4*init_embd_t,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.up4 = Up_C(embed_dim=2*init_embd_t,n_heads=heads,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        
        self.final_up = FinalPatchExpand_X4([64,64],init_embd_t)
        self.output = nn.Conv2d(init_embd_t, 4, kernel_size=1)
                
    def forward(self, inp):
        x0 = self.patch_embd(inp) 
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
                
        f1 = self.up2(x3,x2)        
        f2 = self.up3(f1,x1)        
        f3 = self.up4(f2,x0)    
  
        f4 = self.final_up(f3)
        out = self.output(f4) 
        
        return x1,x2,x3,f1,f2,f3,out

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


    
class Net(nn.Module):
    def __init__(self, n_channels=1):
        super(Net, self).__init__()
        self.n_channels = n_channels
        
        self.student_model = Hybrid_Sudent()
        self.teacher_model = Hybrid_Teacher()
        
        
        # self.ec1 = Conv(init_embd_t,init_embd_s)
        # self.ec2 = Conv(2*init_embd_t,2*init_embd_s)
        # self.ec3 = Conv(4*init_embd_t,4*init_embd_s)
        
        # self.dc1 = Conv(4*init_embd_t,4*init_embd_s)
        # self.dc2 = Conv(2*init_embd_t,2*init_embd_s)
        # self.dc3 = Conv(init_embd_t,init_embd_s)
        
        

    def forward(self, inp):
        x1_s,x2_s,x3_s,f1_s,f2_s,f3_s,out_s = self.student_model(inp)
        x1_t,x2_t,x3_t,f1_t,f2_t,f3_t,out_t = self.teacher_model(inp)
               
        # x1_t = self.ec1(x1_t) 
        # x2_t = self.ec2(x2_t) 
        # x3_t = self.ec3(x3_t)
        
        # f1_t = self.dc1(f1_t)
        # f1_t = self.dc2(f1_t)
        # f3_t = self.dc3(f3_t)

        
        student_features = [x1_s,x2_s,x3_s,f1_s,f2_s,f3_s,out_s] ## 15
        teacher_features = [x1_t,x2_t,x3_t,f1_t,f2_t,f3_t,out_t]
 
        
        # print("Student Model Feature Shapes:")
        # for i, feature in enumerate(student_features):
        #     print(f"Feature {i}: {feature.shape}")
      
        # # Print shapes of teacher features
        # print("Teacher Model Feature Shapes:")
        # for i, feature in enumerate(teacher_features):
        #     print(f"Feature {i}: {feature.shape}")
        
        #return out_s
        return teacher_features,student_features
    
# def model() -> Net:
#     model = Net()
#     model.to(device=DEVICE,dtype=torch.float)
#     return model
# from torchsummary import summary
# model = model()
# summary(model, [(1,256,256)])


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
NUM_WORKERS = 12
PIN_MEMORY=True
DIM_ = 256


def resample_itk_image_LA(itk_image):
    # Get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_spacing = (1,1,1)
    # Calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    # Instantiate resample filter with properties
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    # Execute resampling
    resampled_image = resample.Execute(itk_image)
    return resampled_image
  
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


# Define geometric transformations
geometrical_transforms = tio.OneOf({
    tio.RandomFlip(axes=(1, 2)): 0.5,  # Probability for each geometric transformation
    tio.RandomAffine(degrees=(-45, 45), center='image'): 0.5,
})

# Define intensity transformations
intensity_transforms = tio.OneOf({
    tio.RandomBlur(): 0.25,  # Probability for each intensity transformation
    tio.RandomGamma(log_gamma=(-0.2, -0.2)): 0.25,
    tio.RandomNoise(mean=0.1, std=0.1): 0.25,
    tio.RandomGhosting(axes=(1, 2)): 0.25,
})

# Combine all transformations and probabilities in a single Compose
transforms_2d = tio.Compose([
    tio.OneOf({
        geometrical_transforms: 0.4,  # Probability of applying geometric transformations
        intensity_transforms: 0.4,    # Probability of applying intensity transformations
        tio.Lambda(lambda x: x): 0.2  # Probability of no augmentation
    })
])


   
def generate_label(gt):
        temp_ = np.zeros([4,DIM_,DIM_])
        temp_[0:1,:,:][np.where(gt==0)]=1
        temp_[1:2,:,:][np.where(gt==1)]=1
        temp_[2:3,:,:][np.where(gt==2)]=1
        temp_[3:4,:,:][np.where(gt==3)]=1
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
        img = resample_itk_image_LA(img) 
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_1(img)
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-7]+'_gt.nii.gz'        
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = resample_itk_image_LA(gt) 
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        gt = gt.astype(np.float64)
        
        
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
        img = resample_itk_image_LA(img) 
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_1(img)
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-7]+'_gt.nii.gz'      
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = resample_itk_image_LA(gt) 
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        gt = gt.astype(np.float64)
        
        
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def cal_dice(pre_2d,gt):
    pred = torch.argmax(pre_2d, dim=1)
    
    
    out_LV = torch.zeros_like(pred)
    out_LV[torch.where(pred==1)] = 1
            
    out_MYO = torch.zeros_like(pred)
    out_MYO[torch.where(pred==2)] = 1

    out_RV = torch.zeros_like(pred)
    out_RV[torch.where(pred==3)] = 1
        

    single_LV = (2 * (out_LV * gt[:,1,:]).sum()) / (
               (out_LV + gt[:,1,:]).sum() + 1e-8)
            
            
    single_MYO = (2 * (out_MYO * gt[:,2,:]).sum()) / (
               (out_MYO + gt[:,2,:]).sum() + 1e-8)
            
            
            
    single_RV = (2 * (out_RV * gt[:,3,:]).sum()) / (
               (out_RV + gt[:,3,:]).sum() + 1e-8)
    
    return single_LV,single_MYO,single_RV
            
            
            
def check_Dice_Score(loader, model1, device=DEVICE):
    
    Dice_score_LV_S = 0
    Dice_score_MYO_S = 0
    Dice_score_RV_S = 0

    Dice_score_LV_T = 0
    Dice_score_MYO_T = 0
    Dice_score_RV_T = 0
    
    loop = tqdm(loader)
    model1.eval()
    
    for batch_idx, (img,gt) in enumerate(loop):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
        
        with torch.no_grad(): 
            T,S  = model1(img)
            single_LVs,single_MYOs,single_RVs = cal_dice(S[6],gt)
            
            Dice_score_LV_S +=single_LVs
            Dice_score_MYO_S +=single_MYOs
            Dice_score_RV_S +=single_RVs
            
            single_LVt,single_MYOt,single_RVt = cal_dice(T[6],gt)
            
            Dice_score_LV_T +=single_LVt
            Dice_score_MYO_T +=single_MYOt
            Dice_score_RV_T +=single_RVt

    ## segemntaiton ##
    print(f"Dice_score_LV_S  : {Dice_score_LV_S/len(loader)}")
    print(f"Dice_score_MYO_S  : {Dice_score_MYO_S/len(loader)}")
    print(f"Dice_score_RV_S  : {Dice_score_RV_S/len(loader)}")
    Overall_Dicescore__S = (Dice_score_LV_S + Dice_score_MYO_S + Dice_score_RV_S)/3
    print(f"Overall_Dicescore__S  : {Overall_Dicescore__S/len(loader)}")
    
    print(f"Dice_score_LV_T  : {Dice_score_LV_T/len(loader)}")
    print(f"Dice_score_MYO_T  : {Dice_score_MYO_T/len(loader)}")
    print(f"Dice_score_RV_T  : {Dice_score_RV_T/len(loader)}")
    Overall_Dicescore__TT = (Dice_score_LV_T + Dice_score_MYO_T + Dice_score_RV_T )/3
    print(f"Overall_Dicescore__TT  : {Overall_Dicescore__TT/len(loader)}")

    return Overall_Dicescore__S/len(loader)


class module_1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        max_result,_ = torch.max(x,dim=1,keepdim=True)
        avg_result = torch.mean(x,dim=1,keepdim=True)
        std_result = torch.std(x,dim=1,keepdim=True)
        result = torch.cat([max_result,std_result,avg_result],1)
        return result
def global_min_pooling(feature_map):
    min_layer = feature_map.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    return min_layer.view(feature_map.size(0), -1) 

def global_max_pooling(x):
    max_ = nn.AdaptiveMaxPool2d((1, 1))(x)
    return max_.view(x.size(0), -1) 


def global_std_pooling(x):
    stddev = x.std(dim=[2, 3], keepdim=True)
    return stddev.view(x.size(0), -1)  

def global_avg_pooling(x):
    avg_ = nn.AdaptiveAvgPool2d((1, 1))(x)
    return avg_.view(x.size(0), -1) 

    
class module_2(nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self,x):
        max_result = global_max_pooling(x)
        avg_result = global_avg_pooling(x)
        std_result = global_std_pooling(x)
        min_result = global_min_pooling(x)
        combined = torch.cat([max_result, avg_result, std_result, min_result],1)
        return combined
    
Stats_Ch = module_2()
Stats_F = module_1()

    
import torch
import torch.nn as nn

def norm(x):
    return F.normalize(x, dim=1)
def at_loss(x, y):
    return (norm(x) - norm(y)).pow(2).mean()
    

def top_n_index(std_dev, n):
    batch_size, original_channels = std_dev.shape
    _, top_n_indices = torch.topk(std_dev, n, dim=1)  # Shape: [Batch, n]
    return top_n_indices

def Loss_FT(s1, s2, s3, s4, s5, s6,
            t1, t2, t3, t4, t5, t6):
    
    # List to hold the modified tensors
    t_modified = []

    # List of source tensors for reference
    s_tensors = [s1, s2, s3, s4, s5, s6]
    t_tensors = [t1, t2, t3, t4, t5, t6]
    
    # Iterate over each tensor and apply global_std_pooling and top_n_index
    for s, t in zip(s_tensors, t_tensors):
        t_indices = global_std_pooling(t)
        top_n_indices = top_n_index(t_indices, s.shape[1])  # Use the shape of the current s tensor
        top_n_channels = torch.gather(t, dim=1, index=top_n_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t.shape[2], t.shape[3]))
        t_modified.append(top_n_channels)  # Keep the modified tensor

    # Unpack the modified tensors
    t1, t2, t3, t4, t5, t6 = t_modified
        
    # Calculate losses using the at_loss function
    E1 = at_loss(s1, t1)
    E2 = at_loss(s2, t2)
    E3 = at_loss(s3, t3)
    E4 = at_loss(s4, t4)
    E5 = at_loss(s5, t5)
    E6 = at_loss(s6, t6)

    # Calculate average loss
    Avg = (E1 + E2 + E3 + E4 + E5 + E6) / 6
    return Avg


def Loss_FT_mse(s1, s2, s3, s4, s5, s6,
            t1, t2, t3, t4, t5, t6):
    
    # List to hold the modified tensors
    t_modified = []

    # List of source tensors for reference
    s_tensors = [s1, s2, s3, s4, s5, s6]
    t_tensors = [t1, t2, t3, t4, t5, t6]
    
    # Iterate over each tensor and apply global_std_pooling and top_n_index
    for s, t in zip(s_tensors, t_tensors):
        t_indices = global_std_pooling(t)
        top_n_indices = top_n_index(t_indices, s.shape[1])  # Use the shape of the current s tensor
        top_n_channels = torch.gather(t, dim=1, index=top_n_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t.shape[2], t.shape[3]))
        t_modified.append(top_n_channels)  # Keep the modified tensor

    # Unpack the modified tensors
    t1, t2, t3, t4, t5, t6 = t_modified
        
    # Calculate losses using the at_loss function
    E1 = nn.MSELoss()(s1, t1)
    E2 = nn.MSELoss()(s2, t2)
    E3 = nn.MSELoss()(s3, t3)
    E4 = nn.MSELoss()(s4, t4)
    E5 = nn.MSELoss()(s5, t5)
    E6 = nn.MSELoss()(s6, t6)

    # Calculate average loss
    Avg = (E1 + E2 + E3 + E4 + E5 + E6) / 6
    return Avg

temperature = 4

def KL_Div(student_outputs, teacher_outputs):
    student_log_softmax = F.log_softmax(student_outputs / temperature, dim=1)
    teacher_softmax = F.softmax(teacher_outputs / temperature, dim=1)
    
    distillation_loss = F.kl_div(
        student_log_softmax,
        teacher_softmax,
        reduction='batchmean',
        log_target=False
    ) * (temperature * temperature)
    
    return distillation_loss
    

def Loss_FT_KLD(s1, s2, s3, s4, s5, s6,
            t1, t2, t3, t4, t5, t6):
    
    # List to hold the modified tensors
    t_modified = []

    # List of source tensors for reference
    s_tensors = [s1, s2, s3, s4, s5, s6]
    t_tensors = [t1, t2, t3, t4, t5, t6]
    
    # Iterate over each tensor and apply global_std_pooling and top_n_index
    for s, t in zip(s_tensors, t_tensors):
        t_indices = global_std_pooling(t)
        top_n_indices = top_n_index(t_indices, s.shape[1])  # Use the shape of the current s tensor
        top_n_channels = torch.gather(t, dim=1, index=top_n_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t.shape[2], t.shape[3]))
        t_modified.append(top_n_channels)  # Keep the modified tensor

    # Unpack the modified tensors
    t1, t2, t3, t4, t5, t6 = t_modified
        
    # Calculate losses using the at_loss function
    E1 = KL_Div(s1, t1)
    E2 = KL_Div(s2, t2)
    E3 = KL_Div(s3, t3)
    E4 = KL_Div(s4, t4)
    E5 = KL_Div(s5, t5)
    E6 = KL_Div(s6, t6)

    # Calculate average loss
    Avg = (E1 + E2 + E3 + E4 + E5 + E6) / 6
    return Avg  

def Loss_Mix(s1, s2, s3, s4, s5, s6,
            t1, t2, t3, t4, t5, t6):
    
    # List to hold the modified tensors
    t_modified = []

    # List of source tensors for reference
    s_tensors = [s1, s2, s3, s4, s5, s6]
    t_tensors = [t1, t2, t3, t4, t5, t6]
    
    # Iterate over each tensor and apply global_std_pooling and top_n_index
    for s, t in zip(s_tensors, t_tensors):
        t_indices = global_std_pooling(t)
        top_n_indices = top_n_index(t_indices, s.shape[1])  # Use the shape of the current s tensor
        top_n_channels = torch.gather(t, dim=1, index=top_n_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t.shape[2], t.shape[3]))
        t_modified.append(top_n_channels)  # Keep the modified tensor

    # Unpack the modified tensors
    t1, t2, t3, t4, t5, t6 = t_modified
        
    # Calculate losses using the at_loss function
    E1 = nn.MSELoss()(s1, t1)
    E2 = nn.MSELoss()(s2, t2)
    E3 = KL_Div(s3, t3)
    E4 = nn.MSELoss()(s4, t4)
    E5 = nn.MSELoss()(s5, t5)
    E6 = KL_Div(s6, t6)

    # Calculate average loss
    Avg = (E1 + E2 + E3 + E4 + E5 + E6) / 6
    return Avg  

def Loss_2Combine(s1, s2, s3, s4, s5, s6,
            t1, t2, t3, t4, t5, t6):
    
    # List to hold the modified tensors
    t_modified = []

    # List of source tensors for reference
    s_tensors = [s1, s2, s3, s4, s5, s6]
    t_tensors = [t1, t2, t3, t4, t5, t6]
    
    # Iterate over each tensor and apply global_std_pooling and top_n_index
    for s, t in zip(s_tensors, t_tensors):
        t_indices = global_std_pooling(t)
        top_n_indices = top_n_index(t_indices, s.shape[1])  # Use the shape of the current s tensor
        top_n_channels = torch.gather(t, dim=1, index=top_n_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t.shape[2], t.shape[3]))
        t_modified.append(top_n_channels)  # Keep the modified tensor

    # Unpack the modified tensors
    t1, t2, t3, t4, t5, t6 = t_modified
        
    # Calculate losses using the at_loss function
    E1 = KL_Div(s1, t1)
    E2 = KL_Div(s2, t2)
    E3 = KL_Div(s3, t3)
    E4 = KL_Div(s4, t4)
    E5 = KL_Div(s5, t5)
    E6 = KL_Div(s6, t6)
    # Calculate average loss
    Avg1 = (E1 + E2 + E3 + E4 + E5 + E6) / 6
    
    
    # Calculate losses using the at_loss function
    E1 = nn.MSELoss()(Stats_F(s1), Stats_F(t1))
    E2 = nn.MSELoss()(Stats_F(s2), Stats_F(t2))
    E3 = nn.MSELoss()(Stats_F(s3), Stats_F(t3))
    E4 = nn.MSELoss()(Stats_F(s4), Stats_F(t4))
    E5 = nn.MSELoss()(Stats_F(s5), Stats_F(t5))
    E6 = nn.MSELoss()(Stats_F(s6), Stats_F(t6))
    # Calculate average loss
    Avg2 = (E1 + E2 + E3 + E4 + E5 + E6) / 6
    
    Avg = Avg1 + Avg2

    return Avg  

def train_fn(loader_train1,loader_valid1,model1, optimizer1, scaler1,loss_fn_DC1,epoch): ### Loader_1--> ED and Loader2-->ES

    train_losses1_seg  = [] # loss of each batch
    valid_losses1_seg  = []  # loss of each batch
    
    
    loop = tqdm(loader_train1)
    model1.train()
        
    for batch_idx,(img,gt)  in enumerate(loop):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
       

        with torch.cuda.amp.autocast():
            tc,s  = model1(img)
            
            gt = torch.argmax(gt, dim=1)
            
            out_S = loss_fn_DC1(s[6],gt)
            out_TC = loss_fn_DC1(tc[6],gt)
            
            seg_loss = (out_S + out_TC)   
            
            Loss_feat = Loss_FT_KLD(s[0] ,s[1] ,s[2] ,s[3] ,s[4] ,s[5]  ,tc[0].detach(),tc[1].detach(),tc[2].detach(),tc[3].detach(),tc[4].detach(),tc[5].detach())
            if epoch <50:
                loss = seg_loss 
            else:
                loss = seg_loss  +  Loss_feat
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
           
             tc,s  = model1(img)   
             gt = torch.argmax(gt, dim=1)
             
             out_S = loss_fn_DC1(s[6],gt)
             out_TC = loss_fn_DC1(tc[6],gt)
             
             seg_loss = (out_S + out_TC) 
             
             Loss_feat = Loss_FT_KLD(s[0] ,s[1] ,s[2] ,s[3] ,s[4] ,s[5]  ,tc[0].detach(),tc[1].detach(),tc[2].detach(),tc[3].detach(),tc[4].detach(),tc[5].detach())
             if epoch <50:
                loss = seg_loss
             else:
                loss = seg_loss + Loss_feat
        # backward
        loop_v.set_postfix(loss = loss.item())
        valid_losses1_seg.append(float(loss))

    train_loss_per_epoch1_seg = np.average(train_losses1_seg)
    valid_loss_per_epoch1_seg  = np.average(valid_losses1_seg)
    
    avg_train_losses1_seg.append(train_loss_per_epoch1_seg)
    avg_valid_losses1_seg.append(valid_loss_per_epoch1_seg)    
    
    return train_loss_per_epoch1_seg,valid_loss_per_epoch1_seg

  
#from DC1 import DiceLoss
from Loss3 import DiceCELoss
loss_fn_DC1 = DiceCELoss()


for fold in range(1,2):

  #from m1 import Teacher_Student
  model_1 =  Net()


  fold = str(fold)  ## training fold number 
  
  #train_imgs = r"C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\new_split_mix\F"+fold+"/train/imgs/"
  #val_imgs  = r"C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\new_split_mix\F"+fold+"/val/imgs/"
  
  train_imgs = "/data/scratch/acw676/2D_Seg_12_Sep/MnM2/F"+fold+"/train/imgs/"
  val_imgs  = "/data/scratch/acw676/2D_Seg_12_Sep/MnM2/F"+fold+"/val/imgs/"
  
  Batch_Size = 8
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
    
  path_to_save_check_points = '/data/scratch/acw676/2D_Seg_12_Sep/CNN_1/'+'/F'+fold+'_T'+str(init_embd_t)+'_S'+str(init_embd_s)+'_F'+str(factor)+'Only_CNN_1-Loss_FT_KLD' +'New1'
  path_to_save_Learning_Curve = '/data/scratch/acw676/2D_Seg_12_Sep/CNN_1/'+'/F'+fold+'_T'+str(init_embd_t)+'_S'+str(init_embd_s)+'_F'+str(factor)+'Only_CNN_1-Loss_FT_KLD' +'New1'

  
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
      optimizer1 = torch.optim.AdamW(model1.parameters(),betas=(0.9, 0.99),lr=0.001)
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer1,milestones=[50,100,200,400,500], gamma=0.5)#  milestones=[20,40,60,80,100,120,500], gamma=0.5)
      for epoch in range(Max_Epochs):
          
          train_loss_seg ,valid_loss_seg = train_fn(train_loader,val_loader, model1, optimizer1,scaler1,loss_fn_DC1,epoch)
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
  fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')
