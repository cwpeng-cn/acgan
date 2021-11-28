import torch
import numpy as np
import pylab as plt
from model import Generator
from data import restore_network, recover_image, AnimeDataset

NZ, IMAGE_SIZE, BATCH_SIZE = 100, 64, 8
DEVICE = "cpu"
dataset = AnimeDataset(dataset_path='./anime', image_size=IMAGE_SIZE)
EYES, HAIRS = dataset.EYES, dataset.HAIRS
NUM_EYE, NUM_HAIR = len(EYES), len(HAIRS)

netG = Generator().to(DEVICE)
netG = restore_network("./", "last", netG)

# 眼睛颜色控制
# selected_eye = [5, 5]
# full_image = np.full((len(selected_eye) * IMAGE_SIZE, BATCH_SIZE * IMAGE_SIZE, 3), 0, dtype="uint8")
# fix_noise = torch.randn(BATCH_SIZE, NZ, device=DEVICE)
# fix_input_hair = (torch.rand(BATCH_SIZE, 1) * NUM_HAIR).type(torch.LongTensor).squeeze().to(DEVICE)
# for row, num in enumerate(selected_eye):
#     input_eye = torch.tensor([num for _ in range(BATCH_SIZE)])
#     fake_imgs = netG(fix_noise, input_eye, fix_input_hair).detach().cpu()
#     images = recover_image(fake_imgs)
#     for i in range(BATCH_SIZE):
#         col = i
#         full_image[row * IMAGE_SIZE:(row + 1) * IMAGE_SIZE, col * IMAGE_SIZE:(col + 1) * IMAGE_SIZE, :] = images[i]
# plt.imshow(full_image)
# plt.show()
# plt.imsave("eye.png", full_image)

# 头发颜色控制
# selected_hair = [0, 1, 4, 5]
# full_image = np.full((len(selected_hair) * IMAGE_SIZE, BATCH_SIZE * IMAGE_SIZE, 3), 0, dtype="uint8")
# fix_noise = torch.randn(BATCH_SIZE, NZ, device=DEVICE)
# fix_input_eye = (torch.rand(BATCH_SIZE, 1) * NUM_EYE).type(torch.LongTensor).squeeze().to(DEVICE)
#
# for row, num in enumerate(selected_hair):
#     input_hair = torch.tensor([num for _ in range(BATCH_SIZE)])
#     fake_imgs = netG(fix_noise, fix_input_eye, input_hair).detach().cpu()
#     images = recover_image(fake_imgs)
#     for i in range(BATCH_SIZE):
#         col = i
#         full_image[row * IMAGE_SIZE:(row + 1) * IMAGE_SIZE, col * IMAGE_SIZE:(col + 1) * IMAGE_SIZE, :] = images[i]
#
# plt.imshow(full_image)
# plt.show()
# plt.imsave("hair.png", full_image)

#

fix_noise = torch.randn(2, NZ, device=DEVICE)
fix_input_eye = (torch.rand(2, 1) * NUM_EYE).type(torch.LongTensor).squeeze().to(DEVICE)
input_hair = (torch.rand(2, 1) * NUM_HAIR).type(torch.LongTensor).squeeze().to(DEVICE)

print(input_hair)

for i in range(12):
    input_hair[0] = i
    fake_imgs = netG(fix_noise, fix_input_eye, input_hair).detach().cpu()
    images = recover_image(fake_imgs)
    plt.imshow(images[0])
    plt.pause(2)

# selected_hair = [0]
# full_image = np.full((len(selected_hair) * IMAGE_SIZE, BATCH_SIZE * IMAGE_SIZE, 3), 0, dtype="uint8")
# fix_noise = torch.randn(BATCH_SIZE, NZ, device=DEVICE)
# fix_input_eye = (torch.rand(BATCH_SIZE, 1) * NUM_EYE).type(torch.LongTensor).squeeze().to(DEVICE)
# for row, num in enumerate(selected_hair):
#     input_hair = torch.tensor([0 for _ in range(BATCH_SIZE)])
#     fake_imgs = netG(fix_noise, input_hair, input_hair).detach().cpu()
#     images = recover_image(fake_imgs)
#     for i in range(BATCH_SIZE):
#         col = i
#         full_image[row * IMAGE_SIZE:(row + 1) * IMAGE_SIZE, col * IMAGE_SIZE:(col + 1) * IMAGE_SIZE, :] = images[i]
# plt.imshow(full_image)
# plt.show()
# plt.imsave("eye.png", full_image)
