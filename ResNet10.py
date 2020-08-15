import torch.nn as nn
import torch.nn.functional as F

# 3x3 conv
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)

# residual block
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, kernel_size=3, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_c, out_c, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_c, out_c, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_c)
        #self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        #if self.downsample:
        #    residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet10
class ResNet10(nn.Module):
    def __init__(self, in_resolution=256, out_resolution=256, channel=3, size=16):
        super(ResNet10, self).__init__()
        self.in_resolution = in_resolution
        self.out_resolution = out_resolution
        self.channel = channel
        self.size = size

        # Encoder
        self.block0 = conv3x3(in_channels=3, out_channels=self.size)#256x256x16
        self.block1 = ResBlock(in_c=self.size, out_c=self.size*2, stride=2)#128x128x32
        self.block2 = ResBlock(in_c=self.size*2, out_c=self.size*2, stride=1)#128x128x32
        self.block3 = ResBlock(in_c=self.size*2, out_c=self.size*4, stride=2)#64x64x64
        self.block4 = ResBlock(in_c=self.size*4, out_c=self.size*4, stride=1)#64x64x64
        self.block5 = ResBlock(in_c=self.size*4, out_c=self.size*8, stride=2)#32x32x128
        self.block6 = ResBlock(in_c=self.size*8, out_c=self.size*8, stride=1)#32x32x128
        self.block7 = ResBlock(in_c=self.size*8, out_c=self.size*16, stride=2)#16x16x256
        self.block8 = ResBlock(in_c=self.size*16, out_c=self.size*16, stride=1)#16x16x256
        self.block9 = ResBlock(in_c=self.size*16, out_c=self.size*32, stride=2)#8x8x512
        self.block10 = ResBlock(in_c=self.size*32, out_c=self.size*32, stride=1)#8x8x512

        # Decoder
        self.b1 = nn.ConvTranspose2d(self.size*32, self.size*32, kernel_size=4, stride=1, padding=1)#8x8x512
        self.b2 = nn.ConvTranspose2d(self.size*32, self.size*16, kernel_size=4, stride=2, padding=1)#16x16x256
        self.b3 = nn.ConvTranspose2d(self.size*16, self.size*16, kernel_size=4, stride=1, padding=1)#16x16x256
        self.b4 = nn.ConvTranspose2d(self.size*16, self.size*16, kernel_size=4, stride=1, padding=1)#16x16x256
        self.b5 = nn.ConvTranspose2d(self.size*16, self.size*8, kernel_size=4, stride=2, padding=1)#32x32x128
        self.b6 = nn.ConvTranspose2d(self.size*8, self.size*8, kernel_size=4, stride=1, padding=1)#32x32x128
        self.b7 = nn.ConvTranspose2d(self.size*8, self.size*8, kernel_size=4, stride=1, padding=1)#32x32x128
        self.b8 = nn.ConvTranspose2d(self.size*8, self.size*4, kernel_size=4, stride=2, padding=1)#64x64x64
        self.b9 = nn.ConvTranspose2d(self.size*4, self.size*4, kernel_size=4, stride=1, padding=1)#64x64x64
        self.b10 = nn.ConvTranspose2d(self.size*4, self.size*4, kernel_size=4, stride=1, padding=1)#64x64x64
        self.b11 = nn.ConvTranspose2d(self.size*4, self.size*2, kernel_size=4, stride=2, padding=1)#128x128x32
        self.b12 = nn.ConvTranspose2d(self.size*2, self.size*2, kernel_size=4, stride=1, padding=1)#128x128x32
        self.b13 = nn.ConvTranspose2d(self.size*2, self.size, kernel_size=4, stride=2, padding=1)#256x256x16
        self.b14 = nn.ConvTranspose2d(self.size, self.size, kernel_size=4, stride=1, padding=1)#256x256x16
        self.b15 = nn.ConvTranspose2d(self.size, 3, kernel_size=4, stride=1, padding=1)#256x256x3
        self.b16 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=1, padding=1)#256x256x3
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.block0(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)

        out = self.b1(out)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.b6(out)
        out = self.b7(out)
        out = self.b8(out)
        out = self.b9(out)
        out = self.b10(out)
        out = self.b11(out)
        out = self.b12(out)
        out = self.b13(out)
        out = self.b14(out)
        out = self.b15(out)
        out = self.b16(out)
        out = self.sigmoid(out)

        return out

#net = ResNet10()
#print(net)