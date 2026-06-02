import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from glob import glob

class UniversalSignDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_map = {'mandatory_circle': 0, 'cautionary_triangle': 1, 'informational_square': 2}
        
        for class_name, label_idx in self.class_map.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                continue
            images = []
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
                images.extend(glob(os.path.join(class_folder, ext)))
                
            for img_path in images:
                self.image_paths.append(img_path)
                self.labels.append(label_idx)
                
        print(f"📊 Dataset Loader initialized: Found {len(self.image_paths)} total images across {len(self.class_map)} classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = UniversalSignDataset(root_dir="dataset/weather_damaged", transform=transform_pipeline)

class SimpleSignCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleSignCNN, self).__init__()
        
        self.early_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.middle_layers = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.decision_maker = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.early_layers(x)
        x = self.middle_layers(x)
        x = self.decision_maker(x)
        return x

if len(train_dataset) == 0:
    print("⚠️ Error: Still 0 images found. Let's force-verify your folder structure.")
    if os.path.exists("dataset/weather_damaged"):
        print("Folders inside weather_damaged:", os.listdir("dataset/weather_damaged"))
else:
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    real_model = SimpleSignCNN() 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(real_model.parameters(), lr=0.001)

    print("\nStarting real-data training loop across 5 Epochs...")
    real_model.train()

    for epoch in range(5):
        running_loss = 0.0
        correct_guesses = 0
        total_images = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = real_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            correct_guesses += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (correct_guesses / total_images) * 100
        print(f"Epoch {epoch+1}/5 -> Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.1f}%")

    print("\nTraining complete! Your AI has successfully learned from your uploaded images.")
