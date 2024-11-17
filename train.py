import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import os
import ssl
import urllib.request
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import albumentations as A
import cv2

ssl._create_default_https_context = ssl._create_unverified_context

transform = A.Compose([
    A.Rotate(limit=30, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    A.RandomCrop(width=80, height=80, p=1.0),
])

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply augmentation with fixed random state
    A.ReplayCompose.seed = 42  # Set seed before augmentation
    augmented = transform(image=image)
    augmented_image = augmented['image']
    
    return augmented_image

def show_augmented_images(dataset, num_images=10):
    """Display a grid of original and augmented images"""
    # Create a figure with two rows
    fig, axes = plt.subplots(2, num_images, figsize=(20, 4))
    
    # Get some random training images
    dataiter = iter(torch.utils.data.DataLoader(dataset, batch_size=num_images))
    images, labels = next(dataiter)
    
    # Show original images
    for idx in range(num_images):
        ax = axes[0][idx]
        ax.imshow(images[idx].squeeze(), cmap='gray')
        ax.axis('off')
        if idx == 0:
            ax.set_title('Original', y=-0.2)
    
    # Get another batch for augmented images
    images, labels = next(dataiter)
    
    # Show augmented images
    for idx in range(num_images):
        ax = axes[1][idx]
        ax.imshow(images[idx].squeeze(), cmap='gray')
        ax.axis('off')
        if idx == 0:
            ax.set_title('Augmented', y=-0.2)
    
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    plt.savefig('outputs/augmented_samples.png')
    plt.close()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First Convolutional Block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)  # 28x28 -> 26x26
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # 26x26 -> 24x24
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2)  # 24x24 -> 12x12
        
        # Third Convolutional Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)  # 12x12 -> 10x10
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        
        # Fourth Convolutional Block
        self.conv4 = nn.Conv2d(32, 8, kernel_size=3)  # 10x10 -> 8x8
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 8x8 -> 4x4
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(8 * 4 * 4, 10)  # 4x4x8 -> 10

    def forward(self, x):
        # First Conv Block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        
        # Second Conv Block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool(x)
        
        # Third Conv Block
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        
        # Fourth Conv Block
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 8 * 4 * 4)
        
        # Fully Connected Layers
        x = self.fc1(x)
        return x


def get_device():
    if torch.backends.mps.is_available():
        try:
            # Test MPS availability with a small tensor
            device = torch.device("mps")
            torch.zeros(1).to(device)
            return device
        except:
            print("Warning: MPS (Metal) device found but unusable, falling back to CPU")
            return torch.device("cpu")
    return torch.device("cpu")

def count_parameters(model):
    """Count and format model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_details = {
        'conv1': sum(p.numel() for p in model.conv1.parameters()),
        'conv2': sum(p.numel() for p in model.conv2.parameters()),
        'conv3': sum(p.numel() for p in model.conv3.parameters()),
        'conv4': sum(p.numel() for p in model.conv4.parameters()),
        'fc1': sum(p.numel() for p in model.fc1.parameters()),
        'bn1': sum(p.numel() for p in model.bn1.parameters()),
        'bn2': sum(p.numel() for p in model.bn2.parameters()),
        'bn3': sum(p.numel() for p in model.bn3.parameters()),
        'bn4': sum(p.numel() for p in model.bn4.parameters())
    }
    return total_params, trainable_params, param_details

def train():
    device = get_device()
    print(f"Using device: {device}")
    
    # Define augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Simple transform for visualization
    viz_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    viz_dataset = datasets.MNIST('./data', train=True, download=True, transform=viz_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Show and save augmented images
    print("\nGenerating augmented image samples...")
    show_augmented_images(viz_dataset)
    print("Augmented samples saved to outputs/augmented_samples.png")
    
    # Initialize model
    model = SimpleCNN().to(device)
    
    # Print model parameter details
    total_params, trainable_params, param_details = count_parameters(model)
    print("\nModel Architecture Details:")
    print("="*50)
    print(f"Conv1 parameters: {param_details['conv1']:,}")
    print(f"BatchNorm1 parameters: {param_details['bn1']:,}")
    print(f"Conv2 parameters: {param_details['conv2']:,}")
    print(f"BatchNorm2 parameters: {param_details['bn2']:,}")
    print(f"Conv3 parameters: {param_details['conv3']:,}")
    print(f"BatchNorm3 parameters: {param_details['bn3']:,}")
    print(f"Conv4 parameters: {param_details['conv4']:,}")
    print(f"BatchNorm4 parameters: {param_details['bn4']:,}")
    print(f"FC1 parameters: {param_details['fc1']:,}")
    print("-"*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*50 + "\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train for 1 epoch
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    print("\nStarting training...")
    progress_bar = tqdm(train_loader, desc='Training', unit='batch')
    
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Update progress bar with loss and accuracy
        accuracy = 100 * correct / total
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'accuracy': f'{accuracy:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    final_accuracy = 100 * correct / total
    print(f'\nTraining completed:')
    print(f'Average loss: {avg_loss:.4f}')
    print(f'Training accuracy: {final_accuracy:.2f}%')
    
    # Save model with standardized name
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save current model
    model_path = f'models/mnist_model_latest.pth'
    torch.save({
        'state_dict': model.state_dict(),
        'accuracy': final_accuracy,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }, model_path)
    print(f"Model saved to {model_path}")
    return model

# Debug code to visualize augmentation
def visualize_augmentation(image_path):
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    augmented = transform(image=original)['image']
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(original)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Rotated')
    plt.imshow(augmented)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train() 