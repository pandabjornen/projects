import numpy as np
import torch
from Bio.Align import MultipleSeqAlignment, PairwiseAligner 
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

idx_to_aa = {
    0: 'X', 1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
    12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T',
    18: 'V', 19: 'W', 20: 'Y'
}

aa_to_idx = {v: k for k, v in idx_to_aa.items()}

aa_to_idx['-'] = 0

def num_to_str(seq):
    return ''.join([idx_to_aa.get(x, 'X') for x in seq])

def get_top9(query, others):
    query_str = num_to_str(query)
    scores = []

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1.0 
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0 
    aligner.extend_gap_score = 0.0 

    for s in others:
        
        if np.array_equal(s, query):
            continue
        target_str = num_to_str(s)
        
        
        score = aligner.score(query_str, target_str)
        scores.append((score, s))

    
    scores.sort(reverse=True, key=lambda x: x[0])
    
    return [s[1] for s in scores[:9]]

def msa_tensor(seq_set, max_length):

    str_seqs = [num_to_str(s) for s in seq_set]
    
    padded = [s + '-'*(max_length - len(s)) if len(s) < max_length else s[:max_length] for s in str_seqs]
    
    msa = MultipleSeqAlignment([SeqRecord(Seq(s), id=str(i)) for i, s in enumerate(padded)])
    
    
    
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tensor = torch.zeros(len(msa), max_length, dtype=torch.long, device=device)
    
    
    for i, rec in enumerate(msa):
        for j, aa in enumerate(str(rec.seq)):
            tensor[i, j] = aa_to_idx.get(aa.upper(), 0) 
    return tensor

def build_msa_tensor(all_sequences, max_length_protein):

    n = len(all_sequences)
    

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    output = torch.zeros(n, 10, max_length_protein, dtype=torch.long, device=device)
    
   
    for i, query in enumerate(all_sequences):

        similar = get_top9(query, all_sequences)
        
        msa = msa_tensor([query] + similar, max_length_protein)
        
        output[i] = msa
    return output


