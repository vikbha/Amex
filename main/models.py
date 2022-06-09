from turtle import forward
import torch
import torch.nn as nn
import math



class Vanilla(nn.Module):
    
    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        hidden_dim=64,
        dropout=0.25
        ):
        super().__init__()
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.N1 = nn.LayerNorm(hidden_dim)
        self.R1 = nn.ReLU()
        self.L2 = nn.Linear(hidden_dim, hidden_dim)
        self.N2 = nn.LayerNorm(hidden_dim)
        self.R2 = nn.ReLU()
        self.L3 = nn.Linear(hidden_dim, output_dim)
        self.S3 = nn.Sigmoid()
        
        # Droupout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs):
        x = self.R1(self.N1(self.L1(inputs)))
        x = self.dropout(x)
        x = self.R2(self.N2(self.L2(x)))
        x = self.dropout(x)
        x = self.S3(self.L3(x))
        return x




if __name__=='__main__':
    pass

