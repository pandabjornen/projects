import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
import os
from PIL import Image
from matplotlib.widgets import Button
import numpy as np
import time

def ladda_modellparametrar(modell, mapp, enhet):
    modell_fil = os.path.join(mapp, "modell_params.pth")
    if not os.path.exists(modell_fil):
        raise FileNotFoundError(f"Kunde inte hitta filen: {modell_fil}")
    modell.load_state_dict(torch.load(modell_fil, map_location=enhet))
    modell.to(enhet)
    modell.eval()
    return modell

def återställ(event):
    global prediktioner
    prediktioner.fill(0)
    plt.draw()

def ändra_hastighet(event):
    global hastighetsfaktor
    hastighetsfaktor = 0.5 if hastighetsfaktor == 1 else (0.25 if hastighetsfaktor == 0.5 else 1)
    plt.draw()

def video_test(modell, inställningar):
    mapp = inställningar['Videotestmapp']
    video_sökväg = inställningar['test_video_path']
    klass_etiketter = inställningar['klass_etiketter']
    enhet = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((inställningar['höjd'], inställningar['bredd'])),
        transforms.Grayscale(num_output_channels=inställningar['in_channels']),
        transforms.ToTensor(),
    ])
    
    modell = ladda_modellparametrar(modell, mapp, enhet)
    cap = cv2.VideoCapture(video_sökväg)
    fps = cap.get(cv2.CAP_PROP_FPS)
    global hastighetsfaktor, prediktioner
    hastighetsfaktor = 1
    prediktioner = np.zeros(len(klass_etiketter))
    tidspunkter = []
    frame_tider = []

    tidsfil_sökväg = '../../Tidsmätning/tidsmätningar24.txt'
    with open(tidsfil_sökväg, 'w') as tidsfil:
        plt.ion()
        fig = plt.figure(figsize=(18, 6))  # Ställ in total storlek på din figur
        ax_video = plt.subplot2grid((1, 4), (0, 0),colspan=2)  # Tilldela mer kolumner till videon
        ax_barchart = plt.subplot2grid((1, 4), (0, 2))  # En kolumn till barchart
        ax_scatter = plt.subplot2grid((1, 4), (0, 3)) 
        fig.subplots_adjust(wspace=0.75)  # Justerar mellanrummet mellan subplottarna

        knapp_återställ = plt.axes([0.175, 0.2, 0.1, 0.04])
        btn_reset = Button(knapp_återställ, 'Återställ')
        btn_reset.on_clicked(återställ)

        knapp_hastighet = plt.axes([0.325, 0.2, 0.1, 0.04])
        btn_speed = Button(knapp_hastighet, 'Ändra Hastighet')
        btn_speed.on_clicked(ändra_hastighet)

        while cap.isOpened():
            start_tid = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            bild_interval = int(fps / 10)
            bild = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bild = Image.fromarray(bild)
            bild_tensor = transform(bild).unsqueeze(0).to(enhet)

            with torch.no_grad():
                outputs = modell(bild_tensor)
                _, förutspådd = torch.max(outputs, 1)
                prediktioner[förutspådd.item()] += 1

            ax_video.clear()
            ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax_video.set_title(f'Förutspådd straffriktning: {klass_etiketter[förutspådd.item()]} | Hastighetsfaktor: {hastighetsfaktor}')
            ax_video.axis('off')

            ax_barchart.clear()
            ax_barchart.bar(range(len(klass_etiketter)), prediktioner)
            ax_barchart.set_xlabel('Förutspådd riktning')
            ax_barchart.set_ylabel('Antal förutsägelser')
            ax_barchart.set_ylim([0, max(prediktioner) + 1])
            ax_barchart.set_xticks(range(len(klass_etiketter)))
            ax_barchart.set_xticklabels(['V','M','H'])


            tidspunkt_ms = (time.time() - start_tid) * 1000
            tidspunkter.append(tidspunkt_ms)  # Konverterar tiden till ms
            frame_tider.append(cap.get(cv2.CAP_PROP_POS_FRAMES))

            tidsfil.write(f"{tidspunkt_ms}\n")

            ax_scatter.clear()
            ax_scatter.scatter(frame_tider, tidspunkter)
            ax_scatter.set_xlabel('Bildrutsnummer i videon')
            ax_scatter.set_ylabel('Tid [ms]')
            ax_scatter.set_title('Tid för varje förutsägelse')

            plt.pause(0.01)

            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + bild_interval)

        cap.release()
        plt.ioff()
