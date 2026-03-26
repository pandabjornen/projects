import numpy as np
from Bio.PDB import MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1
import os

def extract_chain_from_cif(paths, output_dir, min_length, max_length, max_ratio_unknown, test_split):
    os.makedirs(output_dir, exist_ok=True)
    paths = [paths] if isinstance(paths, str) else paths
    
    all_coords = []
    all_seqs = []
    valid_paths = []
    
    parser = MMCIFParser(QUIET=True)
    aa_to_index = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWYX')}
    index_to_aa = {i: aa for aa, i in aa_to_index.items()}

    
    for path in paths:
        try:
            structure = parser.get_structure('protein', path)
            chains = list(structure[0].get_chains())
            
            if len(chains) != 1:
                print("too many chains")
                continue
                
            coords, seq_indices = [], []
            for residue in chains[0]:
                if residue.get_id()[0] != ' ' or 'CA' not in residue: continue
                aa = protein_letters_3to1.get(residue.get_resname().title(), 'X')
                coords.append(residue['CA'].get_coord())
                seq_indices.append(aa_to_index[aa] + 1) #for padding ≠0
            if not coords or not seq_indices:
                print("empty coords or sequence")
                continue

            seq_len = len(seq_indices)
            if not (min_length <= seq_len <= max_length) or seq_indices.count(aa_to_index['X'])/seq_len > max_ratio_unknown:
                print("too many unknown")
                continue
            
            all_coords.append(np.array(coords, dtype=np.float32))
            all_seqs.append(np.array(seq_indices, dtype=np.int64))
            valid_paths.append(path)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    if all_coords:
        indices = np.arange(len(all_coords))
        np.random.shuffle(indices)
        split = int(len(indices) * (1 - test_split))
        train_val_idx, test_idx = indices[:split], indices[split:]

        np.save(os.path.join(output_dir, "train_val_coords.npy"), np.array([all_coords[i] for i in train_val_idx], dtype=object))
        np.save(os.path.join(output_dir, "train_val_seqs.npy"), np.array([all_seqs[i] for i in train_val_idx], dtype=object))

        np.save(os.path.join(output_dir, "test_coords.npy"), np.array([all_coords[i] for i in test_idx], dtype=object))
        np.save(os.path.join(output_dir, "test_seqs.npy"), np.array([all_seqs[i] for i in test_idx], dtype=object))

        
    return valid_paths