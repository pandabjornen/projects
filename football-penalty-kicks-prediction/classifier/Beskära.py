import os
from PIL import Image

def beskara_bild(img_path, output_path, inställningar, TEST):
    """
    Gör: Beskär en enskild bild 
    """
    
    vänster_procent = inställningar['vänster_procent']
    höger_procent = inställningar['höger_procent']
    topp_procent = inställningar['topp_procent']
    botten_procent = inställningar['botten_procent']

    if TEST == 'ja':
        vänster_procent = inställningar['test_vänster_procent']
        höger_procent = inställningar['test_höger_procent']
        topp_procent = inställningar['test_topp_procent']
        botten_procent = inställningar['test_botten_procent']


    bild = Image.open(img_path)
    bredd, höjd = bild.size

    vänster = bredd * vänster_procent
    höger = bredd * höger_procent
    topp = höjd * topp_procent
    botten = höjd * botten_procent

    nytt_område = (vänster, topp, bredd - höger, höjd - botten)
    beskuren_bild = bild.crop(nytt_område)

    # Skapa output-mappen om den inte finns
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #Sökvägen för den beskurna bilden
    output_file_path = os.path.join(output_path, os.path.basename(img_path))
    beskuren_bild.save(output_file_path)

def beskara_submappar(input_path, output_path, inställningar, TEST):
    """

    """
    counter = 0
    for subdir, dirs, files in os.walk(input_path):
        # Outputmapp som motsvarar input (H -> H, V->V ...)
        relativ_subdir = os.path.relpath(subdir, input_path)
        output_subdir = os.path.join(output_path, relativ_subdir)
        

        #loopa igenom bilderna:S
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(subdir, file)
                beskara_bild(img_path, output_subdir, inställningar,TEST)
                counter += 1

                #printa var 50 bilder beskurna
                if counter % 50 == 0:
                    print(f"{counter} bilder har beskurits.")

def beskära(input_dir, output_dir, inställningar, TEST):
    
    """

    """

    # Om datatyp 3D går man manuellt en mapp ner 
    if inställningar['datatyp'] == '3D':
        etiketter = ['H', 'V', 'M']
        for etikett in etiketter:
            input_path = os.path.join(input_dir, etikett)
            output_path = os.path.join(output_dir, etikett)
            beskara_submappar(input_path, output_path, inställningar,TEST)
    else:
        beskara_submappar(input_dir, output_dir, inställningar, TEST)

