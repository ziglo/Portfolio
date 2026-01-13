import torch
import torch.nn as nn
import torchvision.models as models

class RegNetClassifier(nn.Module):
    def __init__(self, num_classes, freeze_base=False):
        super(RegNetClassifier, self).__init__()
        
        # 1. Load pretrained RegNet
        self.base_model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        in_features = self.base_model.fc.in_features
        
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

class MoodClassifier:
    def __init__(self, input_size=(3, 224, 224), num_classes=4, freeze_base=False):
        self.model = RegNetClassifier(num_classes, freeze_base=freeze_base)


    def set_freeze_base(self, freeze):
        for param in self.model.base_model.parameters():
            # We only freeze the feature extractor parts, not the fc head
            if param.shape != self.model.base_model.fc[0].weight.shape: # Crude check, usually better to iterate named_params
                 pass 
            

        for param in self.model.base_model.parameters():
            param.requires_grad = not freeze
            

        for param in self.model.base_model.fc.parameters():
            param.requires_grad = True
