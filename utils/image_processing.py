import cv2
import numpy as np
import torch
from torchvision import transforms

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    return image

def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def image_to_tensor(image):
    image = preprocess_image(image)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)  # Add batch dimension

def normalize_image(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return transforms.Normalize(mean=mean, std=std)(image_tensor)