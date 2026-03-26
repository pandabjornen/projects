import torch
import torch.nn as nn

class PositionEncoding(nn.Module): 
    def __init__(self, d_model, maxLengthProtein): #Word embedding dimension and | max_len: maximum len of input and output.  
        super().__init__()

        self.pe = torch.zeros(maxLengthProtein, d_model, device="mps")

        position = torch.arange(start = 0, end = maxLengthProtein, step = 1).float().unsqueeze(1) # unsqueeze(1) -> turned into column matrix (?)
        embedding_index = torch.arange(start = 0, end = d_model, step =  2).float() 

        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, word_embeddings): 

        batch_size, seq_len, d_model = word_embeddings.shape
        # print("word_embeddings shape", word_embeddings.shape)

        positionEncodingsBatchSize = torch.zeros(batch_size, seq_len, d_model, device = "mps")

        for batch in range(batch_size):
            positionEncodingsBatchSize[batch, :, :] = self.pe


        return word_embeddings + positionEncodingsBatchSize