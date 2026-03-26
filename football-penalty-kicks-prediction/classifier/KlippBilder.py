import cv2  #Bildbehandling
import os   # Operativsystem - typ filhantering
import glob # Hitta alla filvägar som uppfyller visst krav


import cv2  #Bildbehandling
import os   # Operativsystem - typ filhantering
import glob # Hitta alla filvägar som uppfyller visst krav



def video_till_bilder(input_path, output_path, extra_steg_bakåt):
    """
    obs : etiketter = ['V', 'M', 'H']
    """
    etiketter = ['V', 'M', 'H']
    counter = 0
    steg_bakåt_ytterligare = extra_steg_bakåt
    for etikett in etiketter:
        mapp_path = os.path.join(input_path, etikett)    #kombinerar för att få sökväg till undermapp
        video_paths = glob.glob(os.path.join(mapp_path, '*.mp4')) + glob.glob(os.path.join(mapp_path, '*.mov'))  #hitta alla .mov och .mp4
    
        print(f"hittade {len(video_paths)} videor i {mapp_path}.")

        output_etikett_path = os.path.join(output_path, etikett) #ut mapp men med rätt etikett (submapp)
        os.makedirs(output_etikett_path, exist_ok=True)  # skapa om inte redan finns

        for video_path in video_paths:    
            cap = cv2.VideoCapture(video_path)  #skapar ngt speciellt objekt som gör att den kan behandlas typ
            if not cap.isOpened():   #gick inte öppna
                print(f"Kunde inte öppna {video_path}")
                continue   #om inte gick öppna video hoppa över resten av koden nedan

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1 - steg_bakåt_ytterligare) # sätter position till sista bild

            # Hade problem innan med andra video till bilder funktioner, enligt chatgpt kanske pga typ att visa frames i slutet var tomma eller ngt,
            # typ att det var problem med .mov. IAF fungerar det nu med nedan kod tror jag, som försöker hitta den sista framen som inte är helt tom. 
            for i in range(total_frames - 1, max(total_frames - 100, -1), -1):  #intererar max 100 frames baklänges
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)  #flyttar positionen till nuvarande frame
                ret, frame = cap.read() # ret är boolean och är true om bilden kunde läsas (return value)
                if ret: # om true ... spara bild
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i-steg_bakåt_ytterligare)
                    ret, frame = cap.read() 
                    frame_path = os.path.join(output_etikett_path, f"{counter:05}.png")   #skapa filnamn  counter för liksom göra snygga namn och snabbt se hur många bilder man har. 
                    cv2.imwrite(frame_path, frame)                                         #spara bilden
                    print(f"Bild sparad som {frame_path}.")
                    counter += 1
                    break
                #else:
                    #print(f"Tom fram @ pos {i} i {video_path}")
            cap.release()



def video_till_bildsekvenser(input_path, output_path):
    """
    Välj FrAMES manuellt i funktionen nedan (BEHÖVER FIXAS): 
    """
    etiketter = ['VU','VN', 'MU','MN', 'HU', 'HN' ]
    counter = 0  
    
    for etikett in etiketter:
        mapp_path = os.path.join(input_path, etikett)
        video_paths = glob.glob(os.path.join(mapp_path, '*.mp4')) + glob.glob(os.path.join(mapp_path, '*.mov'))
        
        print(f"Hittade {len(video_paths)} videor i {mapp_path}.")

        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Kunde inte öppna {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            
            sista_giltiga_frame = total_frames - 1
            for i in reversed(range(total_frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, _ = cap.read()
                if ret:
                    sista_giltiga_frame = i
                    break

            #vilka frames välja ut: 
            frame_ids = [
               
                int(sista_giltiga_frame * 0.8),
                  #procentuellt (av sista giltiga frame) närmaste frame
                int(sista_giltiga_frame * 0.90),
                sista_giltiga_frame
            ]

            
            output_video_path = os.path.join(output_path, etikett, f"bildsekvens{counter}")
            os.makedirs(output_video_path, exist_ok=True)

            for i, frame_id in enumerate(frame_ids):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if ret:
                    frame_path = os.path.join(output_video_path, f"{i}.png")
                    cv2.imwrite(frame_path, frame)
                    print(f"Bild sparad som {frame_path}.")
                else:
                    print(f"Kunde inte läsa frame {frame_id} från {video_path}")

            counter += 1  
            cap.release()


#video_till_bildsekvenser('/Users/andreasmunck/Desktop/Fotbollsstaffar/datan/F_TEST_VIDEOKLIPP', '/Users/andreasmunck/Desktop/Fotbollsstaffar/datan/b_bildsekvenser')