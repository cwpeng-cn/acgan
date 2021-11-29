import os
import torch
import numpy as np
import pylab as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class LossWriter:
    def __init__(self, save_path):
        self.save_path = save_path

    def add(self, loss, i):
        with open(self.save_path, mode="a") as f:
            term = str(i) + " " + str(loss) + "\n"
            f.write(term)
            f.close()


class AnimeDataset(Dataset):

    def __init__(self, dataset_path, image_size):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.EYES = ["blue", "red", "yellow", "green", "purple", "brown"]
        self.HAIRS = ["blonde", "blue", "pink", "purple", "brown", "black"]
        self.img_paths, self.eye_ids, self.hair_ids = self.process(dataset_path)

    def process(self, dataset_path):
        label_path = os.path.join(dataset_path, "label.txt")
        img_paths, eye_ids, hair_ids = [], [], []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                name, eye, hair = line.split('\n')[0].split('\t')
                eye = eye.split(":")[1]
                hair = hair.split(":")[1]
                img_path = os.path.join(dataset_path, "images", name)
                eye_id = self.EYES.index(eye)
                hair_id = self.HAIRS.index(hair)
                img_paths.append(img_path)
                eye_ids.append(eye_id)
                hair_ids.append(hair_id)
        return img_paths, eye_ids, hair_ids

    def __getitem__(self, index):
        data = Image.open(self.img_paths[index])
        image = self.transform(data)
        eye = self.eye_ids[index]
        hair = self.hair_ids[index]
        return image, eye, hair

    def __len__(self):
        return len(self.img_paths)


def recover_image(img):
    return (
            (img.numpy() *
             np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1)) +
             np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1))
             ).transpose(0, 2, 3, 1) * 255
    ).clip(0, 255).astype(np.uint8)


def onehot(label, num_class):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = label.shape[0]
    onehot_label = torch.zeros(n, num_class, dtype=label.dtype).to(device)
    onehot_label = onehot_label.scatter_(1, label.view(n, 1), 1)
    return onehot_label


def save_network(path, network, epoch_label, is_only_parameter=True):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(path, save_filename)
    if is_only_parameter:
        state = network.state_dict()
        for key in state: state[key] = state[key].clone().cpu()
        torch.save(network.state_dict(), save_path, _use_new_zipfile_serialization=False)
    else:
        torch.save(network.cpu(), save_path)


def restore_network(path, epoch, network=None):
    path = os.path.join(path, 'net_%s.pth' % epoch)
    if network is None:
        network = torch.load(path)
    else:
        network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return network


if __name__ == '__main__':
    dataset = AnimeDataset(dataset_path="./anime", image_size=64)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    for i, data in enumerate(data_loader):
        img, eye, hair = data
        img = recover_image(img)[0]
        plt.title("eye:" + dataset.EYES[eye] + "  " + "hair:" + dataset.HAIRS[hair])
        plt.imshow(img)
        plt.pause(0.1)
