import torch  
import torch.nn as nn 
from torchvision import models, transforms, datasets  
from torch.utils.data import DataLoader  
import time 
import matplotlib.pyplot as plt  
import os 
from torchvision.models import resnet18, vgg16, ResNet18_Weights, VGG16_Weights  # Importerar specifika modeller med förtränade vikter
 
def för_träna(inställningar):
    
    original_first_layer = None

    if inställningar['modell_f'] == 'resnet18':

        modell = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Ersätt ett av de sista konvolutionella lagren
        modell.layer4[1].conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        modell.layer4[1].conv2.requires_grad = True

        original_first_layer = modell.conv1  # Hämtar det första lagret 
        modell.conv1 = nn.Conv2d(in_channels=1, out_channels=original_first_layer.out_channels, kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride, padding=original_first_layer.padding)  # Gråskalsanpassar

    elif inställningar['modell_f'] == 'vgg16':

        modell = models.vgg16(weights=VGG16_Weights.DEFAULT) 

        # Ersätt ett av de sista konvolutionella lagren i VGG16
        modell.features[28] = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        modell.features[28].requires_grad = True

        original_first_layer = modell.features[0]  # Hämtar det första lagret i VGG16
        modell.features[0] = nn.Conv2d(in_channels=1, out_channels=original_first_layer.out_channels, kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride, padding=original_first_layer.padding)  # Gråskalsanpassar
        
    # Frys alla lager i modellen så att deras vikter inte uppdateras under träningen
    for param in modell.parameters():
        param.requires_grad = False

    # Anpassar det sista klassificeringslagret för att matcha #klasser
    if inställningar['modell_f'] == 'resnet18':
        num_ftrs = modell.fc.in_features 
        modell.fc = nn.Linear(num_ftrs, inställningar['klasser'])  # Ersätter det fullt anslutna lagret med ett nytt som matchar antalet klasser
        modell.fc.requires_grad = True  # Aktiverar gradientberäkning för det nya lagret
    elif inställningar['modell_f'] == 'vgg16':
        num_ftrs = modell.classifier[6].in_features 
        modell.classifier[6] = nn.Linear(num_ftrs, inställningar['klasser'])  # Ersätter det sista klassificeringslagret med ett nytt som matchar antalet klasser
        modell.classifier[6].requires_grad = True  # Aktiverar gradientberäkning för det nya lagret


    return modell