import os
from PIL import Image
from sklearn.model_selection import train_test_split

class DogMoodDataset:
    def __init__(self, image_dir, transform=None, train_size=0.8, random_state=42):
        self.image_dir = image_dir
        self.transform = transform
        self.train_size = train_size
        self.random_state = random_state
        self.images, self.labels = self.load_data()
        self.label2idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.encoded_labels = [self.label2idx[label] for label in self.labels]
        self.train_images, self.val_images, self.train_labels, self.val_labels = self.split_data()

    def load_data(self):
        images = []
        labels = []
        for label_name in os.listdir(self.image_dir):
            label_path = os.path.join(self.image_dir, label_name)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(label_path, img_file)
                        try:
                            image = Image.open(img_path).convert('RGB')
                            images.append(image)
                            labels.append(label_name)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        return images, labels

    def split_data(self):
        """
        Stratified split to ensure balanced class distribution in train/val sets
        """
        train_images, val_images, train_labels, val_labels = train_test_split(
            self.images,
            self.encoded_labels,
            test_size=(1 - self.train_size),
            random_state=self.random_state,
            stratify=self.encoded_labels  # KEY FIX: ensures balanced split!
        )
        return train_images, val_images, train_labels, val_labels

    def augment_data(self, image):
        # Implement data augmentation techniques
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = self.augment_data(image)
        return image, label