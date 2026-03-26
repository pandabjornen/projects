import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def ladda_modell(modell_sökväg, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(modell_sökväg, map_location=device)) #ladda förtränad modell 
    return model

def testa_modell(dataloader, model, device, class_names, bilder_antal):
    model.to(device) #modell -> cpu eller gpu
    model.eval() #utvärderingsläge -> uppdaterar ej params

    correct = 0
    total = 0
    count = 0  
    #visa bilder
    fig, axs = plt.subplots(bilder_antal // 5, 5, figsize=(15, 3 * (bilder_antal // 5))) if bilder_antal > 0 else None
    axs = axs.flatten() if bilder_antal > 0 else None

    with torch.no_grad():   #inaktivera gradientberäkning -> sparar minne, krävs ej för testning
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)   #flytta etiketter och bilder till gpu eller cpu

            #Förutsägelser: 
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) #ta max av sonnolikhetsfördelningen, 1 står för dim av tensorn. 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #bildvisningsdelen
            if count < bilder_antal:
                for img, label, pred in zip(images, labels, predicted):
                    img = img.cpu().numpy().transpose((1, 2, 0))
                    label = label.cpu().item()
                    pred = pred.cpu().item()

                    if axs is not None and count < len(axs):
                        axs[count].imshow(img[:, :, 0], cmap='gray')
                        axs[count].set_title(f"Korrekt: {class_names[label]}\nFörutsägelse: {class_names[pred]}")
                        axs[count].axis('off')
                        count += 1

    accuracy = 100 * correct / total
    print(f"Testträffsäkerhet: {accuracy:.2f}%")

    if bilder_antal > 0:
        plt.tight_layout()
        plt.show()






def testa(inställningar, modell, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bilder_antal = int(input("Hur många bilder vill du visa? ")) 

    #Torcha: 
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=inställningar['in_channels']),
        transforms.Resize((inställningar['höjd'], inställningar['bredd'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.ImageFolder(root=data, transform=transform)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    class_names = test_dataset.classes  #mappar
    testa_modell(dataloader, modell, device, class_names, bilder_antal)
