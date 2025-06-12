import torch
import torch.nn as nn
import math


class SinusoidalEncoder(nn.Module):
    def __init__(self, input_dim=5, num_frequencies=5):


        super(SinusoidalEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.freq_bands = 2** torch.linspace(0, num_frequencies - 1, num_frequencies)
    
    @property
    def encoding_dim(self):
        return self.input_dim * 2 * self.num_frequencies
 

    def forward(self, x):
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input shape (*, {self.input_dim}), but got {x.shape}")

        x_expanded = x.unsqueeze(-1) * self.freq_bands
        encoded = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1) 
        return encoded.view(x.shape[0], x.shape[1], -1)  # (batch_size, num_rays, coords * 2 * num_frequencies)



#ray2ray
class Ray2Ray(nn.Module):
    def __init__(self, input_dim = 5, output_dim = 4, num_layers=8, hidden_size=256, skip_connect_every=3):
        super(Ray2Ray, self).__init__()
        self.dim_output = output_dim
        self.skip = skip_connect_every
        self.layers = nn.ModuleList()
        self.encoder =SinusoidalEncoder(input_dim=input_dim, num_frequencies=10)
        self.dim_input = self.encoder.encoding_dim
        self.total_params = 0
        self.layers.append(nn.Linear(self.dim_input , hidden_size))

        for i in range(1, num_layers):
            if i % self.skip == 0:
                self.layers.append(nn.Linear(self.dim_input + hidden_size, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.final = nn.Linear(hidden_size, self.dim_output)
        self.act_fn = torch.relu
        self.count_parameters()
        self._initialize_weights()

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        total_megabytes = total_bytes / (1024 ** 2)
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {total_megabytes:.2f} MB")
        self.total_params = total_params
        # exit()
        
        
    def _initialize_weights(self):
        
        # torch.manual_seed(seed)

        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.xavier_uniform_(self.final.weight)
        torch.nn.init.zeros_(self.final.bias)
        
    def forward(self, x):
        x = self.encoder(x)
        x_skip = x
        for i, layer in enumerate(self.layers):
            # print(layer, x.shape)
            if (i % self.skip) == 0 and i > 0:
                x = self.layers[i](torch.cat((x, x_skip), dim=-1))
            else:
                x = self.layers[i](x)
            x = self.act_fn(x)
        x = self.final(x)
        return x


