import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module): 
    def __init__(self, d_model = 2):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = 0 
        self.col_dim = 1

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask = None): 
        
        #Calculate vectors (matrices)

        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)


        #Calculate similarities (sims) between Queries and Keys: 
        sims = torch.matmul(q, k.transpose(-2, -1)) # 3D tensor transpose

        #sims = torch.matmul(q, k.transpose(dim0 = self.row_dim, dim1 = self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5) #Actually only necessary for large models

        if mask is not None: 
            scaled_sims = scaled_sims.masked_fill(mask=mask, value = -1e9)
         
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
