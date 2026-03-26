import os
import shutil
from random import random

def blanda(inställningar):

    data_directory = inställningar['data_directory']
    train_directory = inställningar['train_directory'] 
    val_directory = inställningar['val_directory']

    uppdelning = inställningar['datauppdelning']

    subdirectories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    
    # Skapa tränings- och valideringsmappar om de inte redan finns
    if not os.path.exists(train_directory):
        os.makedirs(train_directory)
        print(f"Skapade träningsmapp: {train_directory}")
    if not os.path.exists(val_directory):
        os.makedirs(val_directory)
        print(f"Skapade valideringsmapp: {val_directory}")
    
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(data_directory, subdirectory)
        train_subdir = os.path.join(train_directory, subdirectory)
        val_subdir = os.path.join(val_directory, subdirectory)
        
        # Skapa motsvarande submappar i tränings- och valideringsmapparna
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)
            print(f"Skapade submapp för träning: {train_subdir}")
        if not os.path.exists(val_subdir):
            os.makedirs(val_subdir)
            print(f"Skapade submapp för validering: {val_subdir}")

        # Räknare för unika filnamn
        image_counter = 1
        
        # Gå igenom varje bildsekvensmapp i submappen
        for sequence in os.listdir(subdirectory_path):
            sequence_path = os.path.join(subdirectory_path, sequence)
            
            if not os.path.isdir(sequence_path):
                continue  # Hoppa över filer som inte är mappar för att inte fastna på BS-saker

            # Bestäm om sekvensen ska gå till träningsmängden eller valideringsmängden med en viss sannolikhet
            target_dir = train_subdir if random() < uppdelning else val_subdir
            
            # Kopiera bilderna direkt till den nya submappen med unika namn
            for image in os.listdir(sequence_path):
                src_image_path = os.path.join(sequence_path, image)
                image_extension = os.path.splitext(image)[1]
                dest_image_name = f"{subdirectory}_{sequence}_{image_counter}{image_extension}"
                dest_image_path = os.path.join(target_dir, dest_image_name)
                shutil.copy(src_image_path, dest_image_path)
                image_counter += 1
            
            print(f"Flyttade bilder från {sequence} till {'träningsmapp' if target_dir == train_subdir else 'valideringsmapp'} i {subdirectory} med unika namn")




