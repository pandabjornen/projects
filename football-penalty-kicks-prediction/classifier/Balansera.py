import os
import random
import shutil
from glob import glob

def balansera(input_dir, output_dir):
    klasser = ['V', 'M', 'H']
    bild_paths = {klass: [] for klass in klasser}

    # Hitta bildpaths för varje klass:
    for klass in klasser:
        klass_path = os.path.join(input_dir, klass)
        bild_paths[klass] = glob(os.path.join(klass_path, '*.png'))
        print(f"Ursprungligt antal bilder i klass '{klass}': {len(bild_paths[klass])}")

    # Bestäm det minsta antalet bilder i någon klass
    min_bilder = min(len(bilder) for bilder in bild_paths.values())
    print(f"Minsta antal bilder i en klass: {min_bilder}")

    # Skapa output-mappen om den inte finns:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Skapade output-mappen: {output_dir}")

    # Kopiera och balansera bilderna
    for klass in klasser:
        output_klass_path = os.path.join(output_dir, klass) 
        os.makedirs(output_klass_path, exist_ok=True)
        
        # Välj slumpmässigt min_bilder från varje klass
        valda_bilder = random.sample(bild_paths[klass], min_bilder)
        
        # Kopiera valda bilder till motsvarande output-mapp
        for bild_path in valda_bilder:
            shutil.copy(bild_path, output_klass_path)
            print(f"Kopierade {bild_path} till {output_klass_path}")
        
    print("Balansering av antalet bilder är klar.")

    # Skriv ut antal bilder i varje klass i output-mappen
    for klass in klasser:
        output_klass_path = os.path.join(output_dir, klass)
        antal_bilder = len(glob(os.path.join(output_klass_path, '*.png')))
        print(f"Antal bilder i klass '{klass}' efter balansering: {antal_bilder}")

