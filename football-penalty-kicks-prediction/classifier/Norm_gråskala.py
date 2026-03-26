from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import numpy as np

class GrayscaleNormalizeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.transform(image)
        return image, label
    

def visa_augmenterade_bilder(dataset, antal_bilder, bilder_per_rad):

    loader = DataLoader(dataset, batch_size=antal_bilder, shuffle=True)
    images, labels = next(iter(loader))

    antal_rader = np.ceil(antal_bilder / bilder_per_rad).astype(int)

    plt.figure(figsize=(bilder_per_rad * 3, antal_rader * 3))
    for i, (image, label) in enumerate(zip(images, labels)):
        image = image.permute(1, 2, 0).numpy()
        plt.subplot(antal_rader, bilder_per_rad, i + 1)
        plt.imshow(image, cmap='grey')
        plt.title(f"Label: {label}")
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

def norm_grå(träningsmängd, valideringsmängd):
    # Applicera gråskalning och normalisering på både tränings- och valideringsdataset
    träningsmängd = GrayscaleNormalizeDataset(träningsmängd)
    valideringsmängd = GrayscaleNormalizeDataset(valideringsmängd)
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Ny total träningsmängd', träningsmängd.__len__(), 'bör vara likadan som den ovan (iaf om inte augment??)')
    print('Ny total valideringsmängd', valideringsmängd.__len__())
    print('följande bör = 0.8',träningsmängd.__len__()/(träningsmängd.__len__()+valideringsmängd.__len__()))
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    visa = input("Vill du visa några av bilderna som skickas in för träning? (ja/nej):")
    if visa.lower() == 'ja':
        antal = int(input("Hur många bilder vill du visa? Ange ett heltal: "))
        bilder_per_rad = int(input("Hur många bilder per rad? Ange ett heltal: "))
        visa_augmenterade_bilder(träningsmängd, antal_bilder=antal, bilder_per_rad=bilder_per_rad)
    return träningsmängd, valideringsmängd



