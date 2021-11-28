from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from data import *
from model import Generator, Discriminator

MODEL_G_PATH = "./"
DATASET_PATH = './anime'
LOG_G_PATH = "Log_G.txt"
LOG_D_PATH = "Log_D.txt"
IMAGE_SIZE = 64
BATCH_SIZE = 128
WORKER = 1
LR = 0.0002
NZ = 100
EPOCH = 150

dataset = AnimeDataset(dataset_path='./anime', image_size=IMAGE_SIZE)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
NUM_EYE = len(dataset.EYES)
NUM_HAIR = len(dataset.HAIRS)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion_bce = nn.BCELoss()
criterion_ce = nn.CrossEntropyLoss()
real_label = 1.
fake_label = 0.
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))

g_writer = LossWriter(save_path=LOG_G_PATH)
d_writer = LossWriter(save_path=LOG_D_PATH)

fix_noise = torch.randn(BATCH_SIZE, NZ, device=device)
fix_input_eye = torch.LongTensor([(i // 5) % NUM_EYE for i in range(BATCH_SIZE)]).squeeze().to(device)
fix_input_hair = torch.LongTensor([(i // 5) % NUM_HAIR for i in range(BATCH_SIZE)]).squeeze().to(device)

print(fix_input_eye[:25])
print(fix_input_hair[:25])

img_list = []
G_losses = []
D_losses = []
iters = 0
loss_weights = [1.5, 0.75, 0.75]

print("开始训练>>>")
for epoch in range(EPOCH):

    print("正在保存网络并评估...")
    save_network(MODEL_G_PATH, netG, epoch)
    with torch.no_grad():
        fake_imgs = netG(fix_noise, fix_input_eye, fix_input_hair).detach().cpu()
        images = recover_image(fake_imgs)
        full_image = np.full((5 * 64, 5 * 64, 3), 0, dtype="uint8")
        for i in range(25):
            row = i // 5
            col = i % 5
            full_image[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64, :] = images[i]
        plt.imshow(full_image)
        plt.imsave("{}.png".format(epoch), full_image)

    for data in data_loader:
        #################################################
        # 1. 更新判别器D: 最大化 log(D(x)) + log(1 - D(G(z)))
        # 等同于最小化 - log(D(x)) - log(1 - D(G(z)))
        #################################################
        netD.zero_grad()
        real_imgs, input_eye, input_hair = data
        input_eye = input_eye.to(device)
        input_hair = input_hair.to(device)
        # 1.1 来自数据集的样本
        real_imgs = real_imgs.to(device)
        b_size = real_imgs.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # 使用鉴别器对数据集样本做判断
        output_d, output_eye, output_hair = netD(real_imgs)
        # 计算交叉熵损失 -log(D(x))
        errD_real = criterion_bce(output_d, label)
        errD_eye = criterion_ce(output_eye, input_eye)
        errD_hair = criterion_ce(output_hair, input_hair)
        errD_real_total = loss_weights[0] * errD_real + loss_weights[1] * errD_eye + loss_weights[2] * errD_hair
        # 对判别器进行梯度回传
        errD_real_total.backward()
        D_x = output_d.mean().item()

        # 1.2 生成随机向量
        noise = torch.randn(b_size, NZ, device=device)
        # 生成随机标签
        input_eye = (torch.rand(BATCH_SIZE, 1) * NUM_EYE).type(torch.LongTensor).squeeze().to(device)
        input_hair = (torch.rand(BATCH_SIZE, 1) * NUM_HAIR).type(torch.LongTensor).squeeze().to(device)
        # 来自生成器生成的样本
        fake = netG(noise, input_eye, input_hair)
        label.fill_(fake_label)
        # 使用鉴别器对生成器生成样本做判断
        output_d, output_eye, output_hair = netD(fake.detach())
        # 计算交叉熵损失 -log(1 - D(G(z)))
        errD_fake = criterion_bce(output_d, label)
        errD_eye = criterion_ce(output_eye, input_eye)
        errD_hair = criterion_ce(output_hair, input_hair)
        errD_fake_total = loss_weights[0] * errD_fake + loss_weights[1] * errD_eye + loss_weights[2] * errD_hair
        # 对判别器进行梯度回传
        errD_fake_total.backward()
        D_G_z1 = output_d.mean().item()

        # 对判别器计算总梯度,-log(D(x))-log(1 - D(G(z)))
        errD = errD_real_total + errD_fake_total
        # 更新判别器
        optimizerD.step()

        #################################################
        # 2. 更新判别器G: 最小化 log(D(x)) + log(1 - D(G(z)))，
        # 等同于最小化log(1 - D(G(z)))，即最小化-log(D(G(z)))
        # 也就等同于最小化-（log(D(G(z)))*1+log(1-D(G(z)))*0）
        # 令生成器样本标签值为1，上式就满足了交叉熵的定义
        #################################################
        netG.zero_grad()
        # 对于生成器训练，令生成器生成的样本为真，
        label.fill_(real_label)
        # 输入生成器的生成的假样本
        output_d, output_eye, output_hair = netD(fake)
        # 对生成器计算损失
        errG = criterion_bce(output_d, label)
        errG_eye = criterion_ce(output_eye, input_eye)
        errG_hair = criterion_ce(output_hair, input_hair)
        errG = loss_weights[0] * errG + loss_weights[1] * errG_eye + loss_weights[2] * errG_hair
        # 对生成器进行梯度回传
        errG.backward()
        D_G_z2 = output_d.mean().item()
        # 更新生成器
        optimizerG.step()

        # 输出损失状态
        if iters % 5 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, EPOCH, iters % len(data_loader), len(data_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            d_writer.add(loss=errD.item(), i=iters)
            g_writer.add(loss=errG.item(), i=iters)

        # 保存损失记录
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1
