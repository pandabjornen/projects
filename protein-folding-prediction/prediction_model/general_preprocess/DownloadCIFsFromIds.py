import os
import requests
def download_mmcif_files(pdb_ids, output_dir) -> list[str]:

    os.makedirs(output_dir, exist_ok=True)
    file_paths = []
    
    for i, pdb_id in enumerate(pdb_ids):
        
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        output_file = os.path.join(output_dir, f"{pdb_id}.cif")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
                
            file_paths.append(output_file)
            print("")
            print(f"Downloading chain {i+1} / {len(pdb_ids)}")
            print("")
            print(f"Downloaded {pdb_id} to {output_file}")
            
            
        except Exception as e:
            print(f"Failed to download {pdb_id}: {e}")
    
    return file_paths




