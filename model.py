import torch
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, num_channel=3, nz=100, neye=11, nhair=12, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入维度 (100+11+12) x 1 x 1
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
        self.apply(weights_init)

    def forward(self, input_z, onehot_eye, onrhot_hair):
        input_ = torch.cat((input_z, onehot_eye, onrhot_hair), dim=1)
        n, c = input_.size()
        input_ = input_.view(n, c, 1, 1)
        return self.main(input_)


class Discriminator(nn.Module):
    def __init__(self, num_channel=3, neye=11, nhair=12, ndf=64, ):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 输入维度 num_channel x 64 x 64
            nn.Conv2d(num_channel, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征维度 (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征维度 (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征维度 (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.discriminator = nn.Sequential(
            # 特征维度 (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.eye_classifier = nn.Conv2d(ndf * 8, neye, 4, 1, 0, bias=False)
        self.hair_classifier = nn.Conv2d(ndf * 8, nhair, 4, 1, 0, bias=False)
        self.softmax = nn.Softmax()

        self.apply(weights_init)

    def forward(self, images):
        feature = self.main(images)
        real_fake = self.discriminator(feature)
        c_eye = self.eye_classifier(feature)
        c_hair = self.hair_classifier(feature)
        real_fake = real_fake.view(-1)
        n = real_fake.shape[0]
        c_eye = c_eye.view(n, -1)
        c_hair = c_hair.view(n, -1)
        return real_fake, c_eye, c_hair


if __name__ == "__main__":
    from data import onehot

    BATCH_SIZE = 8
    NUM_EYE = 11
    NUM_HAIR = 12
    NZ = 100
    device = "cpu"
    input_eye = (torch.rand(BATCH_SIZE, 1) * NUM_EYE).type(torch.LongTensor).squeeze().to(device)
    input_eye = onehot(input_eye, NUM_EYE)
    input_hair = (torch.rand(BATCH_SIZE, 1) * NUM_HAIR).type(torch.LongTensor).squeeze().to(device)
    input_hair = onehot(input_hair, NUM_HAIR)
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    noise = torch.randn(BATCH_SIZE, NZ, device=device)
    images = netG(noise, input_eye, input_hair).detach().cpu()
    print("生成器输出图片尺寸：", images.shape)
    output_d, output_eye, output_hair = netD(images)
    print("判别器输出尺寸：真假:{}，眼睛:{}，头发:{}".format(output_d.shape, output_eye.shape, output_hair.shape))
