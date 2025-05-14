import torch
import numpy as np
import torch.nn as nn
import torch.functional as F


class MixOfModels(nn.Module):
    def __init__(self, input_size, d_model, encoder_num_layers, decoder_num_layers):
        super(MixOfModels, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=512, dropout=0.1, activation='relu', 
                                                   layer_norm_eps=1e-05, batch_first=True, norm_first=True, bias=True, 
                                                   device=None, dtype=None)
        self.embedding_encoder = nn.TransformerEncoder(encoder_layer, num_layers = encoder_num_layers)
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=4, dim_feedforward=512, dropout=0.1, activation='relu', 
        #                                            layer_norm_eps=1e-05, batch_first=True, norm_first=True, bias=True, 
        #                                            device=None, dtype=None)
        self.time_encoder = nn.TransformerDecoder(encoder_layer, num_layers = encoder_num_layers)
        self.duration_encoder = nn.TransformerDecoder(encoder_layer, num_layers = encoder_num_layers)
        self.note_encoder = nn.TransformerDecoder(encoder_layer, num_layers = encoder_num_layers)
        self.volume_encoder = nn.TransformerDecoder(encoder_layer, num_layers = encoder_num_layers)
        self.duration_proj = nn.Linear(d_model, 2)
        self.volume_proj = nn.Linear(d_model, 2)
        self.time_proj = nn.Linear(d_model, 2)
        self.note_proj = nn.Linear(d_model, 128)
        
        
    def forward(self, x):
        x = self.input_proj(x)
        mem = self.embedding_encoder(x)
        print(x.size())
        duration = self.duration_encoder(x)
        volume = self.volume_encoder(x)
        time = self.time_encoder(x)
        note = self.note_encoder(x)
        print(duration.size(), volume.size(), time.size(), note.size())
        duration = self.duration_proj(duration)
        volume = self.volume_proj(volume)
        time = self.time_proj(time)
        note = self.note_proj(note)
        #clip the probabilities to avoid log(0)
        note = torch.clip(note, 0.01, 0.99)
        volume = torch.clip(volume, 0.01, 0.99)
        
        return torch.cat((time, duration, note, volume), dim=1)
    
class Projection(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, note_output = False):
        super(Projection, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        if note_output:
            layers.append(nn.Softmax(dim=1))
        else:
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)