from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
class MonetPhotoDataset(Dataset):
    def __init__(self, root_dir, transform=None, monet=True):
        self.root_dir = os.path.join(root_dir, 'Monet' if monet else 'Photo')
        self.transform = transform
        self.image_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# DataLoaders
monet_dataset = MonetPhotoDataset(root_dir='./data/train', transform=transform, monet=True)
photo_dataset = MonetPhotoDataset(root_dir='./data/train', transform=transform, monet=False)
monet_loader = DataLoader(monet_dataset, batch_size=32, shuffle=True)
photo_loader = DataLoader(photo_dataset, batch_size=32, shuffle=True)

# Downsample and Upsample blocks
def downsample(in_channels, out_channels, apply_instancenorm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    if apply_instancenorm:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)

def upsample(in_channels, out_channels, apply_dropout=False):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels)
    ]
    if apply_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Downsampling stack
        self.down_stack = nn.ModuleList([
            downsample(3, 64, apply_instancenorm=False),  # (batch, 128, 128, 64)
            downsample(64, 128),  # (batch, 64, 64, 128)
            downsample(128, 256),  # (batch, 32, 32, 256)
            downsample(256, 512),  # (batch, 16, 16, 512)
            downsample(512, 512),  # (batch, 8, 8, 512)
            downsample(512, 512),  # (batch, 4, 4, 512)
            downsample(512, 512),  # (batch, 2, 2, 512)
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)  # Bottleneck without InstanceNorm
        ])

        # Upsampling stack
        self.up_stack = nn.ModuleList([
            upsample(512, 512, apply_dropout=True),  # (batch, 2, 2, 1024) after concat
            upsample(1024, 512, apply_dropout=True),  # (batch, 4, 4, 1024) after concat
            upsample(1024, 512, apply_dropout=True),  # (batch, 8, 8, 1024) after concat
            upsample(1024, 512),  # (batch, 16, 16, 1024) after concat
            upsample(1024, 256),  # (batch, 32, 32, 512) after concat
            upsample(512, 128),  # (batch, 64, 64, 256) after concat
            upsample(256, 64),  # (batch, 128, 128, 128) after concat
        ])

        # Final layer
        self.final_layer = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.final_activation = nn.Tanh()  # Output values between -1 and 1

    def forward(self, x):
        skips = []
        
        # Downsampling with skip connections
        for i, down in enumerate(self.down_stack):
            x = down(x)
            skips.append(x)

        skips = skips[:-1][::-1]  # Reverse skips, excluding bottleneck output

        # Upsampling with skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension

        # Final output layer
        x = self.final_layer(x)
        return self.final_activation(x)

# Instantiate the generator
monet_generator = Generator().to(device)

# Function to visualize results
def unnormalize(img):
    img = img * 0.5 + 0.5
    img = img.permute(1, 2, 0)
    return img.cpu().detach().numpy()

def visualize_transformation(generator, data_loader, device, num_images=1):
    generator.eval()
    with torch.no_grad():
        for i, img in enumerate(data_loader):
            if i >= num_images:
                break
            img = img.to(device)
            fake_img = generator(img)
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(unnormalize(img[0]))

            plt.subplot(1, 2, 2)
            plt.title("Generated")
            plt.imshow(unnormalize(fake_img[0]))
            plt.show()

# Visualize a transformation
visualize_transformation(monet_generator, photo_loader, device, num_images=1)
