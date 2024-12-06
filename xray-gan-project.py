import os
import zipfile
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class XRayDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        """
        Custom dataset for loading X-ray images from a zip file
        
        Args:
            zip_path (str): Path to the zip file containing X-ray images
            transform (callable, optional): Optional transform to be applied to images
        """
        self.images = []
        self.transform = transform
        
        # Extract zip file and load images
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for filename in zip_ref.namelist():
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.tiff')):
                    with zip_ref.open(filename) as file:
                        img = plt.imread(file)
                        self.images.append(img)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        if self.transform:
            img = self.transform(img)
        
        return img

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 8, 8)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        validity = self.model(img)
        return validity

def train_gan(zip_path, epochs=100, batch_size=64, lr=0.0002):
    # Data Preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load Dataset
    dataset = XRayDataset(zip_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Networks
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Loss and Optimizers
    adversarial_loss = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Training Loop
    for epoch in range(epochs):
        for real_imgs in dataloader:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # Ground Truths
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            # Train Generator
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, 100).to(device)
            generated_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(generated_imgs), valid)
            g_loss.backward()
            g_optimizer.step()
            
            # Train Discriminator
            d_optimizer.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
        
        # Print Progress
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], "
                  f"D Loss: {d_loss.item():.4f}, "
                  f"G Loss: {g_loss.item():.4f}")
    
    # Save Models
    torch.save(generator.state_dict(), 'xray_generator.pth')
    torch.save(discriminator.state_dict(), 'xray_discriminator.pth')
    
    return generator, discriminator

def main():
    # Replace 'xray_dataset.zip' with your actual zip file path
    zip_path = 'xray_dataset.zip'
    generator, discriminator = train_gan(zip_path)

if __name__ == "__main__":
    main()
