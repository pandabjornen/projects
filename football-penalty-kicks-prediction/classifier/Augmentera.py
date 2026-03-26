from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

class AugmentedDataset(Dataset):
    """
    klicka på Dataset ovan och "go to definition" för att hitta nedan metoder som vi ändrar här för att hantera augmenteringen. 
    Dessa metoder använder sedan DATALOADER när modellen tränas i träna.py
    till exempel när man interar for labels, img in dataloader(): så används __getitem__
    """
    
    def __init__(self, original_dataset, inställningar):
         
        self.original_dataset = original_dataset  
        self.inställningar = inställningar
        self.augment_transform = self._get_augment_transforms() # trnasforms

    def _get_augment_transforms(self):
        """
        man kan ju flytta in hela denna i get item om man vill randomizera för varje bild men försämrar prestanda en del och när jag gjorde detta så överfittade den direkt men kanske bara var otur. s
        """
        transforms_list = []

        #flips:
        # Om Hflip är aktiverad (i inställningar i Kontrollpanel.py) och slumpmässigheten uppfylls, lägg till transformationen.
        if self.inställningar['horisontell_flip?'] and random.random() < self.inställningar['sannolikhet_för_hflip']:
            transforms_list.append(transforms.RandomHorizontalFlip())
        if self.inställningar['vertikal_flip?'] and random.random() < 0.5:
            transforms_list.append(transforms.RandomVerticalFlip())


        #rotation
        if self.inställningar['rotation?'] and random.random() < 0.5:
            transforms_list.append(transforms.RandomRotation(degrees=self.inställningar['rotation_grader']))
        
        #mixa färgen:
        if self.inställningar['color_jitter?'] and random.random() < 0.5:
            transforms_list.append(transforms.ColorJitter(brightness=self.inställningar['ljusstyrka'], 
                                                          contrast=self.inställningar['kontrast'], 
                                                          saturation=self.inställningar['mättnad'], 
                                                          hue=self.inställningar['nyans']))
            
        #Gaussian blur: 
       
        if self.inställningar['gaussian_blur?']and random.random() < 0.5:
            kernel_size = self.inställningar['gaussian_blur_kernel_size']  # example: (5, 5)
            transforms_list.append(transforms.GaussianBlur(kernel_size=kernel_size))

        
        return transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.original_dataset) * 2   # ggr två för att vi kopierar och sen augmenterar kopior
    
    def __getitem__(self, idx):
        """
        Vi har massa index som motsvara varje tensor-grej (bild+label), 
        vi tar och dubblerar antalet index -> kan enkelt kopiera och augmentera bilder i nedan else-sats. 
        
        ( till exempel om vi har index upp till 50 får vi indexen 50, 51, 52 .. som motsvarar 0, 1, 2 ...)
        """
        if idx < len(self.original_dataset):  
            image, label = self.original_dataset[idx]
        else:  
            
            idx -= len(self.original_dataset)
            image, label = self.original_dataset[idx]
            image = self.augment_transform(image)

        return image, label

def augmentera(dataset, inställningar):
    
    augmented_dataset = AugmentedDataset(dataset, inställningar) # ny instans av AugmentedDataset
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Ny total träningsmängd (orginal + augmenterad) =', augmented_dataset.__len__())
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    visa = input("Vill du visa några av de augmenterade bilderna? (ja/nej): ").lower()
    if visa == 'ja':
    
        antal_bilder = int(input("Hur många bilder vill du visa? Ange ett heltal: "))
        bilder_per_rad = int(input("Hur många bilder per rad? Ange ett heltal: "))
        visa_augmenterade_bilder(augmented_dataset, antal_bilder, bilder_per_rad)
    
    return augmented_dataset

def visa_augmenterade_bilder(dataset, antal_bilder, bilder_per_rad):

    loader = DataLoader(dataset, batch_size=antal_bilder, shuffle=True)
    images, labels = next(iter(loader))

    antal_rader = np.ceil(antal_bilder / bilder_per_rad).astype(int)

    plt.figure(figsize=(bilder_per_rad * 3, antal_rader * 3))
    for i, (image, label) in enumerate(zip(images, labels)):
        image = image.permute(1, 2, 0).numpy()
        plt.subplot(antal_rader, bilder_per_rad, i + 1)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()
