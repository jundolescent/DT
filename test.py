import os
from monai.data import ImageReader
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from monai import transforms
from torchvision import datasets
import pandas as pd
import numpy as np
import wfdb
import ast
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import sys
from monai import transforms
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import set_determinism
from sklearn.linear_model import LogisticRegression

from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler

directory = './ecg_images/records100'
# directory = './MEDNIST/'
ECG = ['ecg_I', 'ecg_II', 'ecg_III', 'ecg_AVR', 'ecg_AVL', 'ecg_AVF', 'ecg_V1', 'ecg_V2', 'ecg_V3', 'ecg_V4', 'ecg_V5', 'ecg_V6']


def print_shape(x):
    print(f"Image shape after loading: {x.shape}")
    return x

def to_grayscale(x):
    # 각 채널의 평균을 취하여 흑백 이미지 생성
    if x.shape[0] == 4:  # 만약 채널이 4개라면 알파 채널을 무시
        x = x[:3, :, :]
    grayscale = x.mean(dim=0, keepdim=True)  # 채널 평균을 취해 흑백 이미지 생성
    return grayscale

# Crop (150, 50) ~ (870, 350)
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        # transforms.Lambdad(keys=["image"], func=print_shape),
        transforms.EnsureChannelFirstd(keys=["image"]),
        # transforms.Lambdad(keys=["image"], func=print_shape),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.Lambdad(keys=["image"], func=lambda x: x[0:1, :, :]), # channel 1
        # transforms.Lambdad(keys=["image"], func=to_grayscale),  # 흑백으로 변환
        transforms.EnsureTyped(keys=["image"]),
        transforms.SpatialCropd(keys=["image"], roi_start=[150, 50], roi_end=[870, 350]),
        transforms.Resized(keys=["image"], spatial_size=[64, 64]), # 256, 256
    ]
)

def load_raw_data(df, sampling_rate, path, num_samples=None):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr[:num_samples]]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr[:num_samples]]
    data = np.array([signal for signal, meta in data])
    return data

raw_save_path = f'/home/work/jslee/data/ecg_images/records{100}'

def _transpose_data(data):
    return np.transpose(data)

def transpose_data(data):
    return np.array([_transpose_data(signal) for signal in data])

def load_data(directory=directory, ratio=1, test=False):
    data = []
    limit = int(16601 * ratio) if test else 3
    count = 0
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            if count >= limit:
                break
            count += 1
            # Append each file respectively
            # for file_name in os.listdir(folder_path):
            #     if file_name.endswith('.png'):
            #         file_path = os.path.join(folder_path, file_name)
            #         data.append((file_path, folder))
            # Append folder at once
            data.append((folder_path, folder))
    
    return data
            

class ECGDataset(Dataset):
    def __init__(self, directory, opt_data, transform=None, ratio=1, samplingrate=100, test=False, lead=None): #transform=transforms.ToTensor()
        super(ECGDataset, self).__init__()
        """
        opt_data: 'train', 'validation'
        directory: root directory containing the image folders
        """
        self.opt_data = opt_data
        self.transform = transform
        self.ratio = ratio
        self.test = test
        self.data = load_data(directory, self.ratio, self.test)
        self.samplingrate = samplingrate
        
        if lead == 'all':
            self.lead = ECG
        else:
            self.lead = lead

        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        folderPath, label = self.data[idx]
        if not os.path.isdir(folderPath):
            raise FileNotFoundError(f"Directory not found: {folderPath}")

        label = str(label)
        
        data = {
            "image": folderPath,
            "label": folderPath.split('/')[-1]
        }
        
        if self.transform:
            try:
                # data = self.transform(data)
                temp = []
                for fileName in os.listdir(folderPath):
                    if self.lead == fileName.split('.')[0] or fileName.split('.')[0] == 'ecg_I':
                        file_path = os.path.join(folderPath, fileName)
                        dd = {
                            "image": file_path,
                            "label": fileName.split('.')[0]
                        }
                        ima = self.transform(dd)
                        # np.set_printoptions(threshold=np.inf)
                        temp.append(ima)


                data['image'] = temp
                data['label'] = folderPath.split('/')[-1]
            except Exception as e:
                raise RuntimeError(f"Error applying transform: {e}")
        
        return data["image"], data["label"]
        

dataset = ECGDataset(directory, opt_data='train', transform=train_transforms, ratio=0.1, test=True, lead='ecg_I')
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

def process_batch(batch_idx, images, labels):
    print(f"Batch {batch_idx+1}")
    np.set_printoptions(threshold=np.inf)
    print("Images:", len(images))  # Modified line
    print("Labels:", labels)
    for image in images:
        print(image['label'])
    # with open('temp.txt', 'w') as file:
    #     for image in images[0]['image']:
    #         file.write(np.array2string(image, separator=',', threshold=np.inf) + '\n')


# for batch_idx, (images, labels) in enumerate(train_loader):
#     process_batch(batch_idx, images, labels)


set_determinism(42)

# To-do: 지금 현재 train_loader랑 val_loader랑 같은 데이터를 사용하고 있음. 이를 수정해야함.
dataset = ECGDataset(directory, opt_data='validation', transform=train_transforms, ratio=0.1)
val_loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)


class Diffusion_AE(torch.nn.Module):
    def __init__(self, embedding_dimension = 64):
        super().__init__()
        self.unet = DiffusionModelUNet(
                    spatial_dims=2,
                    in_channels=1,
                    out_channels=1,
                    num_channels=(128, 256, 256),
                    attention_levels=(False, True, True),
                    num_res_blocks=1,
                    num_head_channels=64,
                    with_conditioning=True,
                    cross_attention_dim=1,
                )
        self.semantic_encoder = torchvision.models.resnet18()
        self.semantic_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.semantic_encoder.fc = torch.nn.Linear(512, embedding_dimension)
        
        
    def forward(self, xt, x_cond, t):
        latent = self.semantic_encoder(x_cond)
        noise_pred = self.unet(x=xt, timesteps=t, context=latent.unsqueeze(2))
        return noise_pred, latent
    
device = torch.device("cuda:0")
model = Diffusion_AE(embedding_dimension = 512).to(device)
scheduler = DDIMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
inferer = DiffusionInferer(scheduler)

# parameters
n_iterations = 1e4 # training for longer (1e4 ~ 3h) helps a lot with reconstruction quality, even if the loss is already low
batch_size = 64
val_interval = 100 # 100
iter_loss_list, val_iter_loss_list = [], []
iterations = []
iteration, iter_loss = 0, 0

total_start = time.time()

while iteration < n_iterations:
    for batch_idx, (images, labels) in enumerate(train_loader):
        iteration += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # print(images['label']) # images 자체가 지금 list로 되어있음
        images = images[0]['image']
        # print(len(images))
        # images = torch.stack(images).to(device)
        images = images.to(device)
        noise = torch.randn_like(images).to(device)
        # Create timesteps
        timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (batch_size,)).to(device).long()
        # Get model prediction
        # cross attention expects shape [batch size, sequence length, channels], we are use channels = latent dimension and sequence length = 1
        latent = model.semantic_encoder(images)
        
        # print
        # print(f'Shape of images: {images.shape}')
        # print(f"Shape of noise: {noise.shape}")
        # print(f"Shape of latent: {latent.shape}")
        # print(f"Shape of timesteps: {timesteps.shape}")        
        
        noise_pred = inferer(inputs=images, diffusion_model=model.unet, noise=noise, timesteps=timesteps, condition = latent.unsqueeze(2))
        loss = F.mse_loss(noise_pred.float(), noise.float())

        loss.backward()
        optimizer.step()
        
        iter_loss += loss.item()
        sys.stdout.write(f"Iteration {iteration}/{n_iterations} - train Loss {loss.item():.4f}" + "\r")
        sys.stdout.flush()

        if (iteration) % val_interval == 0:
            model.eval()
            val_iter_loss = 0
            val_step = 0
            for val_step, (images, labels) in enumerate(val_loader):
                with torch.no_grad():
                    images = images[0]['image'].to(device)
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (batch_size,)).to(device).long()
                    noise = torch.randn_like(images).to(device)
                    latent = model.semantic_encoder(images)
                    
                    print(f'Shape of images: {images.shape}')
                    print(f"Shape of noise: {noise.shape}")
                    print(f"Shape of latent: {latent.shape}")
                    print(f"Shape of timesteps: {timesteps.shape}")   
                    noise_pred = inferer(inputs=images, diffusion_model=model.unet, noise=noise, timesteps=timesteps, condition = latent.unsqueeze(2))
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())
                    
                val_iter_loss += val_loss.item()
            iter_loss_list.append(iter_loss / val_interval)
            val_iter_loss_list.append(val_iter_loss / (val_step + 1))
            iterations.append(iteration)
            iter_loss = 0
            print(f"Iteration {iteration} - Interval Loss {iter_loss_list[-1]:.4f}, Interval Loss Val {val_iter_loss_list[-1]:.4f}")


total_time = time.time() - total_start

print(f"train diffusion completed, total time: {total_time}.")

plt.style.use("seaborn-bright")
plt.title("Learning Curves Diffusion Model", fontsize=20)

plt.plot(iterations, iter_loss_list, color="C0", linewidth=2.0, label="Train")
plt.plot(iterations, val_iter_loss_list, color="C4", linewidth=2.0, label="Validation")

plt.yticks(fontsize=12), plt.xticks(fontsize=12)
plt.xlabel("Iterations", fontsize=16), plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.show()
