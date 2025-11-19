import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.mood_classifier import MoodClassifier
from data.dataset import DogMoodDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use the same transform as training validation (no augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
dataset = DogMoodDataset(
    image_dir=r'c:\Users\Austin\Desktop\Projects\dog-mood-detector\src\data',
    transform=None
)

# Load the checkpoint to get label mappings
checkpoint = torch.load("dog_mood_regNet.pth", map_location=device)
label_to_idx = checkpoint['label_to_idx']
idx_to_label = checkpoint['idx_to_label']
num_classes = checkpoint['num_classes']

print(f"\n=== Model Info ===")
print(f"Number of classes: {num_classes}")
print(f"Label mapping: {label_to_idx}")

# Encode labels function (same as train.py)
def encode_labels(labels):
    encoded = []
    for label in labels:
        if isinstance(label, str):
            encoded.append(label_to_idx[label])
        else:
            # Already an integer
            encoded.append(int(label))
    return encoded

val_labels_encoded = encode_labels(dataset.val_labels)

print(f"Validation samples: {len(dataset.val_images)}")
print(f"Sample encoded val labels: {val_labels_encoded[:10]}")

# CustomDataset
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

val_dataset = CustomDataset(dataset.val_images, val_labels_encoded, transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load model
model = MoodClassifier(input_size=(3, 224, 224), num_classes=num_classes).model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\n=== Evaluating ===")

# Track per-class accuracy
class_correct = [0] * num_classes
class_total = [0] * num_classes

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for i in range(len(labels)):
            label_idx = labels[i].item()
            class_total[label_idx] += 1
            if predicted[i] == labels[i]:
                class_correct[label_idx] += 1

accuracy = 100 * correct / total if total > 0 else 0
print(f'\nOverall Validation Accuracy: {accuracy:.2f}%')

# Print per-class accuracy
print(f"\n=== Per-Class Accuracy ===")
for i in range(num_classes):
    if class_total[i] > 0:
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f"{idx_to_label[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"{idx_to_label[i]}: No samples")