import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.mood_classifier import MoodClassifier
from data.dataset import DogMoodDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Define transforms with data augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# 3. Load dataset
dataset = DogMoodDataset(
    image_dir=r'c:\Users\Austin\Desktop\Projects\dog-mood-detector\src\data',
    transform=None
)

# 4. CREATE LABEL MAPPING (STRING -> INTEGER)
unique_labels = sorted(set(dataset.labels))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

print(f"\n=== Dataset Info ===")
print(f"Total samples: {len(dataset.labels)}")
print(f"Label mapping: {label_to_idx}")
print(f"Classes: {unique_labels}")

# DEBUG: Check what type train_labels actually is
print(f"\n=== Debug train_labels ===")
print(f"Type of first train label: {type(dataset.train_labels[0])}")
print(f"First 10 train labels: {dataset.train_labels[:10]}")

# Convert string labels to integers (handle both strings and integers)
def encode_labels(labels):
    encoded = []
    for label in labels:
        if isinstance(label, str):
            encoded.append(label_to_idx[label])
        else:
            # Already an integer
            encoded.append(int(label))
    return encoded

train_labels_encoded = encode_labels(dataset.train_labels)

print(f"Train samples: {len(dataset.train_images)}")
print(f"Sample encoded labels: {train_labels_encoded[:10]}")

# 5. Prepare DataLoader
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = CustomDataset(dataset.train_images, train_labels_encoded, train_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

# 6. Model, loss, optimizer
num_classes = len(unique_labels)
print(f"\n=== Model Info ===")
print(f"Number of classes: {num_classes}")

model = MoodClassifier(input_size=(3, 224, 224), num_classes=num_classes).model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. Training loop with accuracy tracking
num_epochs = 10
print(f"\n=== Training ===")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

# 8. Save model AND label mapping
torch.save({
    'model_state_dict': model.state_dict(),
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'num_classes': num_classes
}, "dog_mood_regNet.pth")

print("\nTraining complete and model saved with label mappings.")
print(f"Label mapping saved: {label_to_idx}")