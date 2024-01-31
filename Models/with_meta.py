import torch
import torch.nn as nn
      
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, padding,stride):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding,stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01 , inplace=True),
        )
    def forward(self, x):
        return self.conv_1(x)
     
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01 , inplace=True),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01 , inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_s2(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01 , inplace=True),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01 , inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_s2 = nn.Sequential(
            DoubleConv_s2(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv_s2(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ClassNet(nn.Module):
    def __init__(self):
        super(ClassNet, self).__init__()
        
        self.m_inc1 = SingleConv(4, 32,3,1,1)
        self.m_inc2 = SingleConv(32, 64,3,1,1)
        self.m_inc3 = SingleConv(64, 128,3,1,2)
        self.m_inc4 = SingleConv(128, 256,3,1,1)
        self.m_inc5 = SingleConv(256, 512,3,1,2)
        self.m_inc6 = SingleConv(512, 1024,3,1,2)
        
        self.linear1 = nn.Linear(4096,128)
        self.linear_v = nn.Linear(128,3)
        self.linear_s = nn.Linear(128,9)
        self.linear_d = nn.Linear(128,6)
        self.linear_f = nn.Linear(128,2)
        
        self.act = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self,Meta_Data):

        m1 = self.m_inc1(Meta_Data)
        m2 = self.m_inc2(m1)
        m3 = self.m_inc3(m2)
        m4 = self.m_inc4(m3)
        m5 = self.m_inc5(m4)
        m6 = self.m_inc6(m5)
        m = torch.flatten(m6,1)
        m = self.act(m)
        m = self.linear1(m)
        logits_v = self.linear_v(m)
        logits_s = self.linear_s(m)
        logits_d = self.linear_d(m)
        logits_f = self.linear_f(m)

        return logits_v,logits_s,logits_d,logits_f
    
class SegNet(nn.Module):
    def __init__(self, n_channels = 1):
        super(SegNet, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        
        self.up0 = Up(1024, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        
        
        self.up0_ = Up(512, 256)
        self.up1_ = Up(256, 128)
        self.up2_ = Up(128, 64)
        self.up3_ = Up(64, 32)
        self.up4_ = Up(64, 32)
        
        self.outc = OutConv(32,2)
        self.outc4 = OutConv(32,4)
        

        self.dropout5E = nn.Dropout2d(p=0.20)  
        self.dropout6E = nn.Dropout2d(p=0.20) 
        
        self.dropout1D = nn.Dropout2d(p=0.20)  
        self.dropout2D = nn.Dropout2d(p=0.20)  


        
    def forward(self, x):

        x1 = self.inc(x)        
        x2 = self.down1(x1)        
        x3 = self.down2(x2)        
        x4 = self.down3(x3)        
        x5 = self.down4(x4) 
        x5 = self.dropout5E(x5)
        x6 = self.down5(x5) 
        x6 = self.dropout6E(x6)
                
        z1 = self.up0(x6, x5)
        z1 = self.dropout1D(z1)
        z2 = self.up1(z1, x4)
        z2 = self.dropout2D(z2)
        z3 = self.up2(z2, x3)
        z4 = self.up3(z3, x2)        
        z5 = self.up4(z4, x1)
        logits1 = self.outc(z5)
        
        y1 = self.up0_(z1, z2)
        y2 = self.up1_(y1, z3)
        y3 = self.up2_(y2, z4)
        y4 = self.up3_(y3, z5)
        
        logits2 = self.outc4(y4)
    
        return logits1,logits2
    
   
class BaseLine3(nn.Module):
    def __init__(self):
        super(BaseLine3, self).__init__()
        self.SegNetwork = SegNet()
        self.ClassNetwork  = ClassNet()

    def forward(self,x,Meta_Data):
        logits_v,logits_s,logits_d,logits_f  = self.ClassNetwork(Meta_Data)
        logits1 , logits2 = self.SegNetwork(x)
        return logits1,logits2,logits_v,logits_s,logits_d,logits_f
  
def model() -> BaseLine3:
    model = BaseLine3()
    return model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torchsummary import summary
model = model()
model.to(device=DEVICE,dtype=torch.float)
summary(model, [(1,256,256),(4,16,16)])

