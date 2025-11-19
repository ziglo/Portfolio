import torch
import torch.nn as nn
import torchvision.models as models

class RegNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RegNetClassifier, self).__init__()
        # Load a pretrained RegNet model
        self.base_model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
        # Replace the classifier head
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

class MoodClassifier:
    def __init__(self, input_size, num_classes):
        self.model = self.build_model(input_size, num_classes)

    def build_model(self, input_size, num_classes):
        model = RegNetClassifier(num_classes)
        return model

    def train(self, train_loader, criterion, optimizer, num_epochs, device):
        self.model.train()
        self.model.to(device)
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path, device):
        self.model.load_state_dict(torch.load(file_path, map_location=device))
        self.model.eval()