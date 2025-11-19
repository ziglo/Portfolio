import torch
from torchvision import transforms
from PIL import Image
import os
from models.mood_classifier import MoodClassifier

class DogMoodPredictor:
    def __init__(self, model_path):
        self.model = MoodClassifier()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
        return predicted.item()

if __name__ == "__main__":
    model_path = os.path.join('models', 'mood_classifier.pth')  # Update with your model path
    predictor = DogMoodPredictor(model_path)
    
    image_path = input("Enter the path to the dog image: ")
    mood = predictor.predict(image_path)
    print(f"The predicted mood of the dog is: {mood}")