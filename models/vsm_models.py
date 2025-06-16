import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import math


class VariationalEncoder(nn.Module):

    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):

        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        

        hidden = self.encoder(x_flat)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        

        mu = mu.view(batch_size, seq_len, self.latent_dim)
        logvar = logvar.view(batch_size, seq_len, self.latent_dim)
        z = z.view(batch_size, seq_len, self.latent_dim)
        
        return mu, logvar, z


class SemanticMemoryModule(nn.Module):

    
    def __init__(self, memory_size, latent_dim, temperature=0.1):
        super(SemanticMemoryModule, self).__init__()
        self.memory_size = memory_size
        self.latent_dim = latent_dim
        self.temperature = temperature
        

        self.memory = nn.Parameter(torch.randn(memory_size, latent_dim))
        nn.init.xavier_uniform_(self.memory)
        

        self.memory_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        
        self.memory_candidate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )
        
    def forward(self, z, update_memory=True):

        batch_size, seq_len, _ = z.shape
        

        # z: [batch_size, seq_len, latent_dim]
        # memory: [memory_size, latent_dim]
        scores = torch.matmul(z, self.memory.t()) / self.temperature  # [batch_size, seq_len, memory_size]
        attention_weights = F.softmax(scores, dim=-1)
        

        attended_memory = torch.matmul(attention_weights, self.memory)  # [batch_size, seq_len, latent_dim]
        

        if update_memory and self.training:
            self._update_memory(z, attention_weights)
        
        return attended_memory, attention_weights
    
    def _update_memory(self, z, attention_weights):

        batch_size, seq_len, _ = z.shape
        

        z_flat = z.view(-1, self.latent_dim)  # [batch_size * seq_len, latent_dim]
        attention_flat = attention_weights.view(-1, self.memory_size)  # [batch_size * seq_len, memory_size]
        

        weighted_input = torch.matmul(attention_flat.t(), z_flat)  # [memory_size, latent_dim]
        attention_sum = attention_flat.sum(dim=0, keepdim=True).t()  # [memory_size, 1]
        
  
        attention_sum = torch.clamp(attention_sum, min=1e-8)
        avg_input = weighted_input / attention_sum  # [memory_size, latent_dim]
        

        concat_input = torch.cat([self.memory, avg_input], dim=-1)  # [memory_size, latent_dim * 2]
        gate = self.memory_gate(concat_input)  # [memory_size, latent_dim]
        candidate = self.memory_candidate(concat_input)  # [memory_size, latent_dim]
        

        self.memory.data = gate * self.memory + (1 - gate) * candidate


class VariationalDecoder(nn.Module):

    
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VariationalDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):

        batch_size, seq_len, _ = z.shape
        z_flat = z.view(-1, self.latent_dim)
        output_flat = self.decoder(z_flat)
        output = output_flat.view(batch_size, seq_len, self.output_dim)
        return output


class VSMModel(nn.Module):

    
    def __init__(self, model_params):
        super(VSMModel, self).__init__()
        self.input_dim = model_params['embed_dim']
        self.hidden_dim = model_params['hidden_size']
        self.latent_dim = model_params.get('latent_dim', 128)
        self.memory_size = model_params.get('memory_size', 512)
        self.beta = model_params.get('beta', 1.0)  # KL损失权重
        self.dropout_ratio = model_params.get('dropout_ratio', 0.1)
        

        self.encoder = VariationalEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )
        

        self.memory = SemanticMemoryModule(
            memory_size=self.memory_size,
            latent_dim=self.latent_dim
        )
        

        self.decoder = VariationalDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim // 4
        )
        

        self.reconstruction_loss = nn.MSELoss()
        
    def forward(self, input, input_len, compute_loss=True):


        mu, logvar, z = self.encoder(input)
        

        attended_memory, attention_weights = self.memory(z)
        

        enhanced_z = z + attended_memory
        

        output = self.decoder(enhanced_z)
        
        losses = {}
        if compute_loss:

            kl_loss = self._compute_kl_loss(mu, logvar, input_len)
            

            recon_loss = self.reconstruction_loss(enhanced_z, z.detach())
            

            memory_reg_loss = self._compute_memory_regularization()
            
            losses = {
                'kl_loss': self.beta * kl_loss,
                'recon_loss': recon_loss,
                'memory_reg_loss': memory_reg_loss,
                'total_loss': self.beta * kl_loss + recon_loss + memory_reg_loss
            }
        
        return output, losses
    
    def _compute_kl_loss(self, mu, logvar, input_len):

        prior = Normal(0, 1)
        posterior = Normal(mu, torch.exp(0.5 * logvar))
        

        kl_div = kl_divergence(posterior, prior)
        

        mask = torch.zeros_like(kl_div[:, :, 0])  # [batch_size, seq_len]
        for i, length in enumerate(input_len):
            mask[i, :length] = 1.0
        

        kl_div_masked = kl_div * mask.unsqueeze(-1)
        kl_loss = kl_div_masked.sum() / mask.sum()
        
        return kl_loss
    
    def _compute_memory_regularization(self):

        memory = self.memory.memory  # [memory_size, latent_dim]
        

        normalized_memory = F.normalize(memory, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_memory, normalized_memory.t())
        

        mask = torch.eye(self.memory_size, device=memory.device)
        similarity_matrix = similarity_matrix * (1 - mask)
        

        reg_loss = similarity_matrix.abs().mean()
        
        return reg_loss
    
    def get_attention_weights(self, input, input_len):

        with torch.no_grad():
            mu, logvar, z = self.encoder(input)
            attended_memory, attention_weights = self.memory(z, update_memory=False)
            return attention_weights 