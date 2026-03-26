#______________________________________________________________________________________________________________________________________________________________________________________________________

from KlippBilder import video_till_bilder, video_till_bildsekvenser
from Beskära import beskära
from Modeller import get_modell
from Träna import träna
from Test import testa, ladda_modell
from Torcha import torcha, torcha2, torcha3D
from Förtränad import för_träna
from Augmentera import augmentera
from Norm_gråskala import norm_grå  
from Blanda import blanda
from Balansera import balansera
from Videotest import video_test

import os
import random
from datetime import datetime
#______________________________________________________________________________________________________________________________________________________________________________________________________



def användarinput(resultat_mapp):
    inställningar = {

        #välj 2D eller 3D: 
        'datatyp': '2D', 

        ##data mappar nedan om "datan" ligger i samma mapp som "Kandidatarbete" ligger i och man står i klassificerar mappen... . Samma sak för modeller 
        #------------------------------------------------------------
        #olika mappar för träning: 
        'videoklipp': '../../datan/A_VIDEOKLIPP',
        'bilder': '../../datan/B_BILDER',

        'balanserade': '../../datan/C_BILDER_BALANSERADE',    


        'beskuren': '../../datan/C_BILDER_BESKURNA',
        'augmenterad': '../../datan/D_BILDER_AUGMENTERADE', 
        'indata': '../../datan/E_BEHANDLADE_BILDER',

        #olika mappar för testning: 
        'test_videoklipp': '../../datan/F_TEST_VIDEOKLIPP',
        'test_bilder':'../../datan/G_TEST_BILDER', 
        'test_bilder_beskurna': '../../datan/H_TEST_BESKURNA', 

        #mappar för 3D: 
        'bildsekvenser': '../../datan/b_bildsekvenser',


        #------------------------------------------------------------

        #Bildsekvenser2D: 

        'data_directory': '../../datan/b_bildsekvenser', 
        'train_directory': '../../datan/träningsmängd' ,
        'val_directory': '../../datan/valideringsmängd',

        'frames': [0.9,0.92,0.94,0.96,0.98,1.00], #antalet frames vid % av klippet
       
        #------------------------------------------------------------
        #Video->bilder:
        #------------------------------------------------------------

        #klippa_träning:
        'steg_bakåt_extra_från_sista_frame': 0,

        #klippa_test:
        'TEST_steg_bakåt_extra_från_sista_frame': 0,

        #Beskära (% i decimalform) , nedan värden till fifa: 
        'vänster_procent': 0.375, 
        'höger_procent': 0.3, 
        'topp_procent': 0.09, 
        'botten_procent': 0.225, 
        
        #beskära test:
        'test_vänster_procent': 0, 
        'test_höger_procent': 0, 
        'test_topp_procent': 0, 
        'test_botten_procent': 0, 
        #------------------------------------------------------------
        ###Augmentera: 
        #------------------------------------------------------------
        #flippa
        'horisontell_flip?': True,  #True/False
        'sannolikhet_för_hflip': 0.5, #random.random() < detta tal -> sätt = 1 för alltid hflippa sätt 0.5 för hflipp 50%. 
        #transforms.RandomHorizontalFlip()

        'vertikal_flip?': False,  #True/False
        #transforms.RandomHorizontalFlip()

        #rotera
        'rotation?': True,           #True/False
        'rotation_grader':  10, 

        #variation av färger typ: 
        'color_jitter?': True,       #True/False
        #förändringar 
        'ljusstyrka': 0.5, 
        'kontrast': 0.5, 
        'mättnad': 0.5, 
        'nyans':0.2, 



        #Gaussian blur: 
        'gaussian_blur?': True,
        'gaussian_blur_kernel_size': (3,3),  

        #------------------------------------------------------------
        ##modell arkitektur:
        #------------------------------------------------------------
        'modell': 'TinyVGG', 
        'stride': 1,
        'padding': 1,
        'noder': 7,
        'klasser': 3, 
        'höjd': 256, #används även i torcha
        'bredd': 256, #används även i torcha
        'in_channels': 1,  
        #OBSOBS tror att om nedan ändras måste även Modeller.py modifieras eftersom filterstorleken påverkar ngt... 
        'filter_storlek': (3,3),  #(x, y) x är antalet rader och y antalet kolumner, till exempel (3,3). (eller vet inte ordningen men antar det). 
        'filter_storlek_maxpool': (2,2), 

        #3D: 
        '3Dmodell': 'modell3D',

        'höjd_3D': 256, 
        'bredd_3D': 256, 
        'djup_3D': 6, 

        'filter_storlek_conv_3D': (3,3,3), 
        'filter_storlek_maxpool_3D': (2,2,2), 

        'stride_3D': (1,1,1), 
        'stride_mp_3D': (2,2,2), 
        #------------------------------------------------------------

        ###träning:
        #------------------------------------------------------------ 
        'temperatur': 0.5, 
        'datauppdelning': 0.8,
        'batch_size': 32,
        'inlärningstakt': 1e-3,
        'epochs': 40,
        'optimizer_choice': 'Adam',
        #spara: 
        'modell_save_path': os.path.join(resultat_mapp, 'modell_params.pth'),   #OBS glöm ej ändra för att spara modell (EJ RADERA TIDIGARE VERSIONER)
        'resultat_mapp': resultat_mapp,
        'träningsplott': os.path.join(resultat_mapp,'träningsplott'),           #obs ändra
        'träningsstatistik': os.path.join(resultat_mapp, 'träningsstatestik'),    #obs ändra   ||||| stäng ovan plott för att det ska sparas tydligen :)

        #förträning: 
        'modell_f': 'resnet18', 
        #------------------------------------------------------------
        ##TEST
        #------------------------------------------------------------
        # vanligt 2D test:

        'Test_modell_path': '../../Resultat/4.1/modell_params_epok_21_TS_60.19_VK_2.09.pth', #obs ändra

        # videotest: 

        'klass_etiketter': {0: "Vänster", 1: "Mitten", 2: "Höger"},  #kanske höger och vänster byta plats
        'test_video_path': '/Users/andreasmunck/Desktop/testvideo5.mp4', 
        'Videotestmapp': '/Users/andreasmunck/Desktop/Fotbollsstaffar/Resultat/till video test', 
        #------------------------------------------------------------
    }

    print("Valda inställningar:")
    for key, value in inställningar.items():
        print(f"{key}: {value}")
    
    return inställningar

def spara_inställningar_till_fil(inställningar, resultat_mapp):
    inställningar_filnamn = os.path.join(resultat_mapp, 'inställningar.txt')
    with open(inställningar_filnamn, 'w') as f: 
        for key, value in inställningar.items():
            f.write(f"{key}: {value}\n")

def huvudfunktion():

    #TESTA innan träning:
    VIDEOTEST = input('Videotest (direkt)? (ja/nej):').lower()
    TEST2d = input('Testa 2D (direkt)? (ja/nej):').lower()
   

    ###skapa mapp för resultat: 

    #skapa mapp med namnet av datum
    namn_val = input("Vill du använda nuvarande tid som mappnamn? (ja/nej): ").lower()

    if namn_val == 'ja':     
        mappnamn = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        mappnamn = input("Ange ett namn för mappen: ")

    resultat_mapp = os.path.join('../../Resultat', mappnamn)

    print(f"Resultat sparas i'{resultat_mapp}'")
    os.makedirs(resultat_mapp, exist_ok=True) 

    inställningar = användarinput(resultat_mapp)
    
    spara_inställningar_till_fil(inställningar, resultat_mapp)
    #TEST direkt__________________________________________________________________________________________________________________________________
    if VIDEOTEST == 'ja': 
        modell = get_modell(inställningar['modell'], inställningar)
        video_test(modell, inställningar)

    
    if TEST2d == 'ja': 
        modell = get_modell(inställningar['modell'], inställningar)
        data = inställningar['test_bilder']
        print('---'*20)
        print('OBS du testar nu:')
        print('---'*20)
        TEST_KLIPP = input(f'KLIPPA video -> bilder (OBS extra steg bakåt från sista frame: {inställningar["TEST_steg_bakåt_extra_från_sista_frame"]} )? (ja/nej):').lower()
        if TEST_KLIPP == 'ja': 
            video_till_bilder(inställningar['test_videoklipp'], inställningar['test_bilder'], inställningar['TEST_steg_bakåt_extra_från_sista_frame'])
        TEST_BESKÄRA = input('BESKÄRA? (ja/nej):').lower()
        if TEST_BESKÄRA == 'ja': 
            beskära(data, inställningar['test_bilder_beskurna'], inställningar,'ja')
            data = inställningar['test_bilder_beskurna']
        modell1 = ladda_modell(inställningar['Test_modell_path'] ,modell)    
        testa(inställningar, modell1, data)    
    #__________________________________________________________________________________________________________________________________

    #träna:


    datatyp_val = input("3D eller 2D? (2D/3D): " ).lower().strip()
    if datatyp_val == '2d':
        modell = get_modell(inställningar['modell'], inställningar)

        BORT_MED_ALLA_INPUTFRÅGOR = input('Vill du bara köra? (ja/nej):').lower()
        

        data  = inställningar['videoklipp']

        #databehandling: 
        
        if BORT_MED_ALLA_INPUTFRÅGOR == 'nej':
            KLIPP_BILDER = input('Vill du klipp->bilder? (ja/nej): ').lower()
            if KLIPP_BILDER == 'ja':
                print("Videoklipp -> bilder...")
                video_till_bilder(data, inställningar['bilder'], inställningar['steg_bakåt_extra_från_sista_frame']) 
                data = inställningar['bilder']

            else:
                data = inställningar['bilder']


            BALANSERA = input('Vill du balansera datan (lika många bilder i varje klass)? (ja/nej):').lower()
            if BALANSERA == 'ja':
                balansera(data, inställningar['balanserade'])
                data = inställningar['balanserade']
        


            BESKÄRA = input('Vill du beskara? (ja/nej): ').lower()
            if BESKÄRA == 'ja':     
                print("Beskär...")
                beskära(data, inställningar['beskuren'], inställningar, 'nej')
                data = inställningar['beskuren']

        if BORT_MED_ALLA_INPUTFRÅGOR == 'ja':
            data = inställningar['bilder'] #ändra till det du vill, två vanligt använda är: 'bilder' och 'beskuren'.  
            
        print('--------------')
        print(f'Datan som skickas in i torcha kommer från {data}')
        print('--------------')

        BILDSEKVENSER2D = input('Vill du använda flera bilder från samma klipp? (ja/nej):').lower()
        if BILDSEKVENSER2D == 'ja': 

            #blanda(inställningar)  #kommentera bort om du redan gjort en gång
            träningsmängd, valideringsmängd = torcha2(trän_dir= inställningar['train_directory'], val_dir = inställningar['val_directory'], inställningar=inställningar)

        else: 
            print("Torchar...")
            träningsmängd, valideringsmängd = torcha(data, inställningar) 
        
        
        AUGMENTERA = input('Vill du augmentera? (ja/nej): ').lower()
        if AUGMENTERA == 'ja': 
            träningsmängd = augmentera(träningsmängd, inställningar)
        
        träningsmängd, valideringsmängd = norm_grå(träningsmängd, valideringsmängd)
        
        ###
        print("Dataförberedelser klara.")
        ###

        if BORT_MED_ALLA_INPUTFRÅGOR == 'nej':
            FÖRTRÄNA = input('Vill du använda en förtränad modell? (ja/nej):').lower()
        
            if FÖRTRÄNA == 'ja': 
                modell = för_träna(inställningar)

        print("Börjar träning:")
        print('__________________________________________________________________________________________________________________________________')

        träna(träningsmängd,valideringsmängd,modell,inställningar, datatyp_val)
        
        print("Tränat klart -> test")
        #NEDAN FUNGERAR EJ!!!
        TEST = input('Testa FUNKAR EJ? (ja/nej):').lower()
        if TEST == 'ja': 
            data = inställningar['test_bilder']
            TEST_KLIPP = input('KLIPPA? (ja/nej):').lower()
            if TEST_KLIPP == 'ja': 
                video_till_bilder(inställningar['test_videoklipp'], inställningar['test_bilder'])
            TEST_BESKÄRA = input('BESKÄRA? (ja/nej):').lower()
            if TEST_BESKÄRA == 'ja': 
                beskära(data, inställningar['test_bilder_beskurna'], inställningar,'ja')
                data = inställningar['test_bilder_beskurna']
            testa(inställningar, modell, data)
    elif datatyp_val == '3d': #3D
        
        modell = get_modell(inställningar['3Dmodell'], inställningar)

        träningsmängd, valideringsmängd = torcha3D(inställningar['bildsekvenser'], inställningar)

        print('börjar träning: ')
        träna(träningsmängd, valideringsmängd, modell, inställningar, datatyp_val)

    else: 
        raise ValueError('välj 2D eller 3D')
if __name__ == '__main__':
    huvudfunktion()
