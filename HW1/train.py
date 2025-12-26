import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LeNet5(nn.Module):
    def __init__(self, activation='sigmoid'):
        super(LeNet5, self).__init__()
        
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError("Activation must be 'sigmoid' or 'relu'")

        # Layer 1: Conv2d (1 -> 6, 5x5), Output: 28x28 (因為 Input 是 32x32)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        
        # Layer 2: AvgPool2d (2x2), Output: 14x14 
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Conv2d (6 -> 16, 5x5), Output: 10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Layer 4: AvgPool2d (2x2), Output: 5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Layer 5: Conv2d (16 -> 120, 5x5), Output: 1x1
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        
        # Layer 6: Linear (120 -> 84)
        self.fc1 = nn.Linear(120, 84)
        
        # Layer 7: Linear (84 -> 10) Output
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # C1 -> Act -> S2
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool1(x)
        
        # C3 -> Act -> S4
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool2(x)
        
        # C5 -> Act
        x = self.conv3(x)
        x = self.act(x)
        # Flatten
        x = torch.flatten(x, 1)
        
        # F6 -> Act
        x = self.fc1(x)
        x = self.act(x)
        
        # Output
        x = self.fc2(x)
        return x

def get_dataloaders(batch_size=32):  
    # 1. Resize to 32x32
    # 2. Data Augmentation (Flip, Rotation)
    # 3. ToTensor & Normalize
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.RandomHorizontalFlip(),  
        # transforms.RandomVerticalFlip(),    
        transforms.RandomRotation(15),      
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def plot_results(train_acc, val_acc, train_loss, val_loss, activation_name):
    epochs = range(len(train_acc))
    
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Valid Loss') # 
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Valid Acc')
    plt.title('Accuracy Curve') # 
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    
    # 存檔 
    filename = f"Loss&Acc_{activation_name}.jpg"
    plt.suptitle(f'Training with {activation_name}', fontsize=16) # 
    plt.savefig(filename)
    plt.close()
    print(f"Figure saved as {filename}")

def train_model(activation_name, epochs=20): # At least 15 epochs 
    print(f"\nExample: Training LeNet-5 with {activation_name} activation...")
    
    model = LeNet5(activation=activation_name.lower()).to(device)
     
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader, val_loader = get_dataloaders()
    
    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []
    best_acc = 0.0
 
    if not os.path.exists('./model'):
        os.makedirs('./model')

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_train / total_train
        train_loss_hist.append(epoch_loss)
        train_acc_hist.append(epoch_acc)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct_val / total_val
        val_loss_hist.append(epoch_val_loss)
        val_acc_hist.append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
 
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            save_path = f"./model/Weight_{activation_name}.pth" # 
            torch.save(model.state_dict(), save_path)
            # print(f"Best model saved to {save_path}")

    plot_results(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist, activation_name)
    print(f"Training {activation_name} finished. Best Val Acc: {best_acc:.2f}%")

if __name__ == '__main__':
    # 1. Train with Sigmoid
    train_model('Sigmoid', epochs=20) 
    
    # 2. Train with ReLU
    train_model('Relu', epochs=20)