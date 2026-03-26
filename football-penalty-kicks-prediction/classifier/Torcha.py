import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset,random_split
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def _torcha_(data_dir, inställningar):
   


    # gråskala, ändra storlek, omvandla till tensorer (så att pytroch kan behandla dem), 
    transform = transforms.Compose([   
        transforms.Resize((inställningar['höjd'], inställningar['bredd'])),
        transforms.ToTensor(),
    ])

   

    full_datamängd = datasets.ImageFolder(root=data_dir, transform=transform)   #ladda datamängd från en mappstruktur (t.ex H, M och V submappar. varje mapp får en etikett (klass))
    
    #blanda datan:
    train_size = int(inställningar['datauppdelning'] * len(full_datamängd))
    indices = torch.randperm(len(full_datamängd)).tolist()  
    train_dataset = Subset(full_datamängd, indices[:train_size])
    val_dataset = Subset(full_datamängd, indices[train_size:])

    return train_dataset, val_dataset

def _torcha_2(trän_dir, val_dir, inställningar):
   


    # gråskala, ändra storlek, omvandla till tensorer (så att pytroch kan behandla dem), 
    transform = transforms.Compose([   
        transforms.Resize((inställningar['höjd'], inställningar['bredd'])),
        transforms.ToTensor(),
    ])

   

    träning = datasets.ImageFolder(root=trän_dir, transform=transform)   #ladda datamängd från en mappstruktur (t.ex H, M och V submappar. varje mapp får en etikett (klass))
    validering = datasets.ImageFolder(root=val_dir, transform=transform)

    trän_dataset = Subset(träning, range(len(träning)))
    val_dataset = Subset(validering, range(len(validering)))

    return trän_dataset, val_dataset

def visa_bilder_med_filnamn(dataset, antal, bilder_per_rad):
   
    antal_rader = antal // bilder_per_rad + (1 if antal % bilder_per_rad else 0)
    plt.figure(figsize=(15, 2 * antal_rader))
    for i in range(antal):
        img, label = dataset[i]
        img_path = dataset.dataset.imgs[dataset.indices[i]][0]
        img_name = os.path.basename(img_path)
        
        ax = plt.subplot(antal_rader, bilder_per_rad, i + 1)
        img = img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f'{img_name} : label= {label}', fontsize=5)  
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def torcha(data_dir, inställningar):
    """
    Input:
        data_dir (str): Sökvägen till datamappen.
        inställningar (dict): En dictionary med inställningar för bildförberedelse.

    Gör:
        Anropar _torcha_ för att förbereda train_dataset och val_dataset baserat på angivna inställningar.
        Frågar om  några bearbetade bilder från train_dataset skall visas.
    
    Hur:
        Använder inställningar för att definiera transformeringar och dela upp dataset i tränings- och valideringsset.
        Använder ImageFolder och Subset från torchvision för att ladda och dela upp datan.

    Output:
        train_dataset (Subset): En datamängd för träning.
        val_dataset (Subset): En datamängd för validering.
    """
    train_dataset, val_dataset = _torcha_(data_dir, inställningar)
    
    
    visa = input("Vill du visa några av de bearbetade bilderna? [OBS: Stäng plotten av bilder för att starta träningen!!!]   (ja/nej):")
    if visa.lower() == 'ja':
        antal = int(input("Hur många bilder vill du visa? Ange ett heltal: "))
        bilder_per_rad = int(input("Hur många bilder per rad? Ange ett heltal: "))
        visa_bilder_med_filnamn(train_dataset, antal=antal, bilder_per_rad=bilder_per_rad)

    return train_dataset, val_dataset


def torcha2(trän_dir, val_dir, inställningar):
    train_dataset, val_dataset = _torcha_2(trän_dir, val_dir, inställningar)
    
    
    visa = input("Vill du visa några av de bearbetade bilderna? [OBS: Stäng plotten av bilder för att starta träningen!!!]   (ja/nej):")
    if visa.lower() == 'ja':
        antal = int(input("Hur många bilder vill du visa? Ange ett heltal: "))
        bilder_per_rad = int(input("Hur många bilder per rad? Ange ett heltal: "))
        visa_bilder_med_filnamn(train_dataset, antal=antal, bilder_per_rad=bilder_per_rad)

    return train_dataset, val_dataset


#3DD::::::




class BildSekvensDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.sekvenser = self._hitta_sekvenser(root_dir)
    
    def _hitta_sekvenser(self, root_dir):
        sekvenser = []
        for klass in ['V', 'M', 'H']:
            klass_path = os.path.join(root_dir, klass)
            for sekvens_mapp in os.listdir(klass_path):
                sekvens_path = os.path.join(klass_path, sekvens_mapp)
                if os.path.isdir(sekvens_path):
                    sekvenser.append((sekvens_path, klass))
        return sekvenser
    
    def __len__(self):
        return len(self.sekvenser)
    
    def __getitem__(self, idx):
        sekvens_path, klass = self.sekvenser[idx]
        bilder = []
        for bild_fil in sorted(os.listdir(sekvens_path)):
            bild_path = os.path.join(sekvens_path, bild_fil)
            bild = Image.open(bild_path).convert('RGB')
            if self.transform:
                bild = self.transform(bild)
            bilder.append(bild)
        bilder_tensor = torch.stack(bilder)
        klass_idx = {'V': 0, 'M': 1, 'H': 2}[klass]  # Konvertera klass till en index. Anpassa detta vid behov.
        return bilder_tensor, klass_idx



def torcha3D(data_dir, inställningar):
   
    transform = transforms.Compose([
        transforms.Resize((inställningar['höjd'], inställningar['bredd'])),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),      #ta bort om augmenterar
        transforms.Normalize(mean=[0.5], std=[0.5])       # ta bort om augmenterar
        
    ])

    dataset = BildSekvensDataset(root_dir=data_dir, transform=transform)
    train_size = int(inställningar['datauppdelning'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    visa = input("Vill du visa några av de bearbetade bildsekvenserna? [OBS: Stäng plotten av bilder för att starta träningen!!!] (ja/nej): ").lower()
    if visa == 'ja':
        antal_sekvenser = int(input("Hur många bildsekvenser vill du visa? Ange ett heltal: "))
        visa_bildsekvenser(train_dataset, antal_sekvenser)

    return train_dataset, val_dataset

def visa_bildsekvenser(dataset, antal):
    
    plt.figure(figsize=(20, 5 * antal))  
    
    for i in range(antal):
        idx = np.random.randint(0, len(dataset))
        sekvens_tensor, _ = dataset[idx]
        
       
        antal_bilder = sekvens_tensor.size(0)
        
       
        for j in range(antal_bilder):
            plt.subplot(antal, antal_bilder, i * antal_bilder + j + 1)  
            plt.imshow(sekvens_tensor[j].squeeze(), cmap='gray')  
            plt.axis('off')
        
    plt.tight_layout()
    plt.show()