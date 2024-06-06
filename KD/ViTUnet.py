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

init_embd = 24
img_size = 160
patch_size = 2

class PatchEmbed(nn.Module): # [2,1,160,160] -->[2,1600,96]
    def __init__(self, img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=init_embd, norm_layer=nn.LayerNorm):
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
        self.down = nn.Sequential(
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
            PatchMerging(embed_dim)
        )
    def forward(self, x):
        return self.down(x)
    
class Up(nn.Module):
    def __init__(self, embed_dim,n_heads,mlp_ratio,qkv_bias,p,attn_p,up):
        super().__init__()
        
        self.up = PatchExpand(embed_dim)
        self.blk = nn.Sequential(
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
                    attn_p=attn_p,up=up)
                                 )
        
    def forward(self, x2,x1):
        
        x2 = self.up(x2)
        c =  torch.cat((x1, x2),1)
    
        return self.blk(c)
    
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.patch_embd  = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=1,
                embed_dim=init_embd,
        )
        self.down1 = Down(embed_dim=init_embd,n_heads=4,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        
        self.down2 = Down(embed_dim=2*init_embd,n_heads=4,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.down3 = Down(embed_dim=4*init_embd,n_heads=4,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        self.down4 = Down(embed_dim=8*init_embd,n_heads=4,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.)
        
        self.up1 = Up(embed_dim=16*init_embd,n_heads=4,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.,up=True)
        self.up2 = Up(embed_dim=8*init_embd,n_heads=4,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.,up=True)
        self.up3 = Up(embed_dim=4*init_embd,n_heads=4,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.,up=True)
        self.up4 = Up(embed_dim=2*init_embd,n_heads=4,mlp_ratio=4., qkv_bias=True,p=0.,attn_p=0.,up=True)
        
        self.final_up = PatchExpand_final2(init_embd)
        self.output = nn.Conv2d(init_embd, 5, kernel_size=1, bias=False)
                
    def forward(self, inp):
        x0 = self.patch_embd(inp)        
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        y = self.up1(x4,x3)
        y = self.up2(y,x2)        
        y = self.up3(y,x1)        
        y = self.up4(y,x0)        
        y = self.final_up(y)
        y = self.output(y)    
        return y
    

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def model() -> Net:
    model = Net()
    model.to(device=DEVICE,dtype=torch.float)
    return model
from torchsummary import summary
model = model()
summary(model, [(1,160,160)])
