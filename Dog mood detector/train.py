import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
from models.mood_classifier import MoodClassifier
from data.dataset import DogMoodDataset

# seeding.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# set device. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#trainer.
train_transform = transforms.Compose([
    transforms.Resize((256, 256)), # Resize first
    transforms.RandomCrop(224),    # Then crop (keeps more context than ResizedCrop)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = DogMoodDataset(
    image_dir=r'c:\Users\Austin\Desktop\Projects\dog-mood-detector\dogproject\data',
    transform=None
)

# map labels.
unique_labels = sorted(set(dataset.labels))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(unique_labels)

print(f"Classes: {unique_labels}")

def encode_labels(labels):
    encoded = []
    for label in labels:
        if isinstance(label, str):
            encoded.append(label_to_idx[label])
        else:
            encoded.append(int(label))
    return encoded

train_labels_encoded = encode_labels(dataset.train_labels)

# load the dataset. 
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform: img = self.transform(img)
        return img, label

train_dataset = CustomDataset(dataset.train_images, train_labels_encoded, train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

# intialize model
print("\n=== Initializing Model ===")
# We start unfrozen immediately but use a smart scheduler
classifier = MoodClassifier(input_size=(3, 224, 224), num_classes=num_classes, freeze_base=False)
model = classifier.model.to(device)

# label smoothing, tells the model if it looks 90% sad and 10% neutral it is okay, prevents model from being too overconfident. 
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=30)

# Training Loop
def run_epoch(optimizer, model, loader, epoch_idx):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step() # Step per batch for OneCycleLR
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

print("\n=== Training with Label Smoothing & Context Preserving ===")
num_epochs = 30

for epoch in range(num_epochs):
    loss, acc = run_epoch(optimizer, model, train_loader, epoch)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | Loss: {loss:.4f} | Acc: {acc:.2f}%")


torch.save({
    'model_state_dict': model.state_dict(),
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'num_classes': num_classes
}, "dog_mood_regNet_v3.pth")

print("\nTraining complete. Model saved as dog_mood_regNet_v3.pth")
