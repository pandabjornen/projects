import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
import time
import os

def beräkna_klassvikter(dataset):
    """
    Beräknar vikter för varje klass baserat på deras frekvens i datasetet.
    Använder inversen av klassfrekvenserna som vikter och skriver ut dessa vikter.
    """
    class_counts = {}
    for _, target in dataset:
        if target in class_counts:
            class_counts[target] += 1
        else:
            class_counts[target] = 1
    
    # Antal olika klasser
    num_classes = len(class_counts)
    print(f"Antal olika klasser: {num_classes}")
    
    # Beräkna vikterna
    total = sum(class_counts.values())
    weights = torch.tensor([total/class_counts[i] if i in class_counts else 0 for i in range(num_classes)], dtype=torch.float32)
    
    # Normalisera vikterna
    weights /= weights.sum()
    
    # Skriv ut klassernas frekvenser och deras respektive vikter
    for i, count in class_counts.items():
        print(f"Klass {i}: Frekvens = {count}, Vikt = {weights[i]:.4f}")
    
    return weights


def välj_optimerare(modell, inlärningstakt, val_av_optimerare):
    """
    Input:
        modell: Modellen som ska tränas.
        inlärningstakt: Inlärningshastigheten för optimizern.
        optimizer_choice: Valet av optimizer ('Adam' eller 'SGD').
    Gör:
    Hur: 
    Output:
        En optimizer för den angivna modellen.
    """
    ## KRÄVS FÖR FÖRTRÄNA: 
    optimizer_parameters = filter(lambda p: p.requires_grad, modell.parameters())  # Filtrerar modellparametrar som kräver gradienter för att användas av optimeraren
    ###
    if val_av_optimerare == 'Adam':
        return optim.Adam(optimizer_parameters, lr=inlärningstakt)
    elif val_av_optimerare == 'SGD':
        return optim.SGD(optimizer_parameters, lr=inlärningstakt, momentum=0.9)
    else:
        print("Ogiltigt val/ej inlagt i Träna.py -> använder Adam.")
        return optim.Adam(optimizer_parameters, lr=inlärningstakt)

def träna(train_dataset, val_dataset, modell, inställningar, datatyp):
    """ 
    Input:
        train_dataset: Datamängden för träning.
        val_dataset: Datamängden för validering.
        modell: Modellen som ska tränas.
        inställningar: Dictionary med träningsinställningar.
    Gör:
        Tränar modellen med angivet tränings- och valideringsdataset.

    Hur: Grad_descent i varje epoch...  Dataloader ... ??

    Output:
        Tränad modell + info och graf av träning. 
    """
    temperatur = inställningar['temperatur']
    train_loader = DataLoader(train_dataset, batch_size=inställningar['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=inställningar['batch_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modell.to(device)
    optimizer = välj_optimerare(modell, inställningar['inlärningstakt'], inställningar['optimizer_choice'])
    
    weights = beräkna_klassvikter(train_dataset)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device)) 
    #loss_fn = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    start_tid = time.time()  # Starttid för hela träningen
    print('')
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print(f" Epok  |        Träningskostnad       |       Träningsträffsäkerhet [%] |     Valideringskostnad    |     Valideringsträffsäkerhet [%] |    Epok tid [s]   |     Tid kvar [min]  |")
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    
    for epoch in range(inställningar['epochs']):
        epoch_start_time = time.time()  # Starttid för nuvarande epok

        modell.train()
        total_train_loss, total_train_correct, total_train = 0, 0, 0

        for inputs, targets in train_loader:

            if datatyp == '3d': 
                inputs = inputs.permute(0,2,1,3,4)  #byt plats på kanaler och djup
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = modell(inputs)
            scaled_logits = outputs / temperatur  # Skala logits med angiven temperatur
            loss = loss_fn(scaled_logits, targets)  # Beräkna förlust med skalade logits
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train_correct += (predicted == targets).sum().item()
            total_train += targets.size(0)

        train_losses.append(total_train_loss / len(train_loader)) 
        train_accuracies.append(100*total_train_correct / total_train) ###

        modell.eval()
        total_val_loss, total_val_correct, total_val = 0, 0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                if datatyp == '3d': 
                    inputs = inputs.permute(0,2,1,3,4)  #byt plats på kanaler och djup
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = modell(inputs)
                loss = loss_fn(outputs, targets)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val_correct += (predicted == targets).sum().item()
                total_val += targets.size(0)

        val_losses.append(total_val_loss / len(val_loader))
        val_accuracies.append(100*total_val_correct / total_val) ###

        epoch_end_time = time.time()  # Sluttid för nuvarande epok
        epoch_duration = epoch_end_time - epoch_start_time  # Tid det tog att köra nuvarande epok
        est_tid = epoch_duration*inställningar['epochs']/60   # Hur långt baserat på nuvarande epok som den totala tiden väntas vara
        total_tid_kvar = est_tid-(epoch_end_time-start_tid)/60 # hur långt tills slut
        print(f" {epoch+1}/{inställningar['epochs']}  |             {train_losses[-1]:.4f}            |             {train_accuracies[-1]:.4f}             |             {val_losses[-1]:.4f}          |             {val_accuracies[-1]:.4f}              |       {epoch_duration:.2f}       |          {total_tid_kvar:.2f}      | ")
        
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

        #spara varje epok
        torch.save(modell.state_dict(), os.path.join(inställningar['resultat_mapp'], f'modell_params_epok_{epoch}_TS_{round(100*total_val_correct / total_val,2)}_VK_{round(total_val_loss / len(val_loader),2)}.pth'))

    total_duration = time.time() - start_tid  # Total tid för hela träningen
    print(f"Total träningstid: {total_duration:.2f} s")

    # Spara modellen
    torch.save(modell.state_dict(), inställningar['modell_save_path'])###

   #plott
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, inställningar['epochs']+1), train_losses, label='Träningskostnad')
    plt.plot(range(1, inställningar['epochs']+1), val_losses, label='Valideringskostnad')
    plt.xlabel('Epok')
    plt.ylabel('Kostnadsfunktionen')
    plt.title('Tränings- och valideringsförlust')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, inställningar['epochs']+1), train_accuracies, label='Träningsträffsäkerhet')
    plt.plot(range(1, inställningar['epochs']+1), val_accuracies, label='Valideringsträffsäkerhet')
    plt.xlabel('Epok')
    plt.ylabel('Träffsäkerhet [%]')
    plt.title('Träning och valideringsträffsäkerhet')
    plt.legend()

    plt.tight_layout()

    #spara plotten som pngbild
    sökväg = inställningar['träningsplott']
    plt.savefig(sökväg)
    print(f"Träningsgraf sparad till: {sökväg}")

    plt.show()

    #spara träningsstatestikdatan som txt.fil
    listor_sökväg = inställningar['träningsstatistik']
    with open(listor_sökväg, 'w') as f:
        f.write("träningskostnadsfunktion:\n")
        f.write(str(train_losses) + "\n")
        f.write("Valideringskostnadsfunktion:\n")
        f.write(str(val_losses) + "\n")
        f.write("Träningsträffsäkerheten:\n")
        f.write(str(train_accuracies) + "\n")
        f.write("Valideringsträffsäkerheten:\n")
        f.write(str(val_accuracies))
    print(f"sparat till: {listor_sökväg}")
