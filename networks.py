from fastai.conv_learner import *
from fastai.dataset import *
from bam import *

class SEModule(nn.Module):
    def __init__(self, ch, re=16):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch,ch//re,1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re,ch,1),
                                 nn.Sigmoid()
                               )
    def forward(self, x):
        return x * self.se(x)
    
class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch,ch//re,1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re,ch,1),
                                 nn.Sigmoid()
                               )
        self.sSE = nn.Sequential(nn.Conv2d(ch,ch,1),
                                 nn.Sigmoid())
        
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
    
class GCN(nn.Module):
    def __init__(self,inp,out,ks=7):
        super().__init__()
        self.conv_l = nn.Sequential(nn.Conv2d(inp,out,(ks,1),padding=(ks//2,0)),
                                   nn.Conv2d(out,out,(1,ks),padding=(0,ks//2))
                                  )
        self.conv_r = nn.Sequential(nn.Conv2d(inp,out,(1,ks),padding=(0,ks//2)),
                                   nn.Conv2d(out,out,(ks,1),padding=(ks//2,0))
                                  )
    def forward(self,x):
        return self.conv_l(x) + self.conv_r(x)
    
class Refine(nn.Module):
    def __init__(self,inp):
        super().__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(inp),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inp,inp,3,padding=1),
                                   nn.BatchNorm2d(inp),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inp,inp,3,padding=1))
    def forward(self,x):
        return x + self.conv1(x)

def conv_block(ni,nf,ks = 3):
    model = nn.Sequential(nn.Conv2d(ni,ni,kernel_size=ks,padding=ks//2),nn.ReLU(inplace=True),nn.BatchNorm2d(ni),
                        nn.Conv2d(ni,nf,kernel_size=ks,padding=ks//2))
    return model

def create_interpolate(x,img_size,mode='bilinear',ac=False):
    sf = img_size//x.size(2) 
    return x if sf == 1 else F.interpolate(x,scale_factor=sf,align_corners=ac,mode=mode) 

def unet_conv(ni,nf): 
    model = nn.Sequential(nn.ReLU(),nn.BatchNorm2d(ni),
                       nn.Conv2d(ni,ni,3,padding=1),
                       nn.ReLU(),nn.BatchNorm2d(ni),
                       nn.Conv2d(ni,nf,3,padding=1))
    return model

class ResNetWithBAM(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = torchvision.models.resnet34(True)
        encoder.conv1.stride = (1,1)
        self.input_adjust = nn.Sequential(encoder.conv1,
                                  encoder.bn1,
                                  encoder.relu)
        self.pool = encoder.maxpool
        self.conv1 = encoder.layer1
        self.conv2 = encoder.layer2
        self.conv3 = encoder.layer3
        self.conv4 = encoder.layer4
        self.bam1,self.bam2,self.bam3,self.bam4 = BAM(64),BAM(128),BAM(256),BAM(512)
        
    def forward(self,x):
        inp =self.input_adjust(x) 
        e0 = self.pool(inp)
        e1 = self.bam1(self.conv1(e0))
        e2 = self.bam2(self.conv2(e1))
        e3 = self.bam3(self.conv3(e2))
        e4 = self.bam4(self.conv4(e3))
        return e0,e1,e2,e3,e4

class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.bam = BAM(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bam(self.bn(F.relu(cat_p)))