import torch
from torch import nn
from data import onehot
from model import Generator
from data import restore_network

NZ, IMAGE_SIZE, BATCH_SIZE = 100, 64, 6
DEVICE = "cpu"
NUM_EYE, NUM_HAIR = 6, 6


class Generator(nn.Module):
    def __init__(self, num_channel=3, nz=100, neye=6, nhair=6, ngf=64):
        super(Generator, self).__init__()
        self.neye = neye
        self.nhair = nhair

        self.main = nn.Sequential(
            # 输入维度 (100+6+6) x 1 x 1
            nn.ConvTranspose2d(nz + neye + nhair, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 特征维度 (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 特征维度 (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 特征维度 (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 特征维度 (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, num_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # 特征维度. (num_channel) x 64 x 64
        )

    def forward(self, concat_input):
        input_z, eye, hair = concat_input[:, :100], concat_input[:, 100], concat_input[:, 101]
        eye = eye.type(torch.int64)
        hair = hair.type(torch.int64)
        eye = onehot(eye, self.neye)
        hair = onehot(hair, self.nhair)
        input_ = torch.cat((input_z, eye, hair), dim=1)
        n, c = input_.size()
        input_ = input_.view(n, c, 1, 1)
        return self.main(input_)


netG = Generator().to(DEVICE)
netG = restore_network("./", "acgan_generator", netG)

x = torch.randn(1, 102)
traced_script_module = torch.jit.trace(func=netG, example_inputs=x)
traced_script_module.save("ACGAN_Generator.pt")
