"""
Auto-Encodeur Variationnel Récurrent (VAE-RNN) pour la génération de séquences d'écriture manuscrite.

Architecture:
- Encodeur RNN/LSTM: encode les trajectoires d'écriture en représentation latente
- Espace latent: distribution gaussienne pour contrôle du style
- Décodeur RNN/LSTM: génère de nouvelles trajectoires à partir de l'espace latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    """
    Encodeur RNN qui transforme les séquences d'écriture manuscrite
    en représentation latente.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, latent_dim, dropout=0.1):
        """
        Args:
            input_dim: Dimension des entrées (ex: 2 pour x, y ou 3 pour x, y, pen_state)
            hidden_dim: Dimension de l'état caché LSTM
            num_layers: Nombre de couches LSTM empilées
            latent_dim: Dimension de l'espace latent
            dropout: Taux de dropout
        """
        super(EncoderRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # Couche LSTM pour encoder la séquence
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Couches pour calculer la moyenne et la variance de l'espace latent
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: Tensor de forme (batch_size, seq_len, input_dim)
        
        Returns:
            mu: Moyenne de la distribution latente (batch_size, latent_dim)
            logvar: Log-variance de la distribution latente (batch_size, latent_dim)
            hidden: État caché final du LSTM
        """
        # Encoder la séquence avec LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Utiliser le dernier état caché (dernière couche)
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Calculer la moyenne et la variance de l'espace latent
        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)
        
        return mu, logvar, (hidden, cell)


class DecoderRNN(nn.Module):
    """
    Décodeur RNN qui génère des séquences d'écriture manuscrite
    à partir de l'espace latent.
    """
    
    def __init__(self, latent_dim, hidden_dim, num_layers, output_dim, max_seq_len, dropout=0.1):
        """
        Args:
            latent_dim: Dimension de l'espace latent
            hidden_dim: Dimension de l'état caché LSTM
            num_layers: Nombre de couches LSTM empilées
            output_dim: Dimension des sorties (ex: 2 pour x, y ou 3 pour x, y, pen_state)
            max_seq_len: Longueur maximale des séquences à générer
            dropout: Taux de dropout
        """
        super(DecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        
        # Projeter l'espace latent vers l'état initial du LSTM
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        # Couche LSTM pour décoder la séquence
        self.lstm = nn.LSTM(
            output_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Couche de sortie pour prédire les coordonnées
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def init_hidden(self, z):
        """
        Initialise l'état caché du LSTM à partir du vecteur latent.
        
        Args:
            z: Vecteur latent (batch_size, latent_dim)
        
        Returns:
            hidden: État caché initial (num_layers, batch_size, hidden_dim)
            cell: État de cellule initial (num_layers, batch_size, hidden_dim)
        """
        batch_size = z.size(0)
        
        # Projeter le vecteur latent vers les états initiaux
        hidden = self.latent_to_hidden(z)
        cell = self.latent_to_cell(z)
        
        # Reshape pour correspondre à la forme attendue par LSTM
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.transpose(0, 1).contiguous()
        
        cell = cell.view(batch_size, self.num_layers, self.hidden_dim)
        cell = cell.transpose(0, 1).contiguous()
        
        return hidden, cell
    
    def forward(self, z, target_seq=None, teacher_forcing_ratio=0.5):
        """
        Génère une séquence à partir du vecteur latent.
        
        Args:
            z: Vecteur latent (batch_size, latent_dim)
            target_seq: Séquence cible pour teacher forcing (batch_size, seq_len, output_dim)
            teacher_forcing_ratio: Probabilité d'utiliser teacher forcing
        
        Returns:
            output: Séquence générée (batch_size, seq_len, output_dim)
        """
        batch_size = z.size(0)
        device = z.device
        
        # Initialiser l'état caché à partir du vecteur latent
        hidden, cell = self.init_hidden(z)
        
        # Déterminer la longueur de séquence
        if target_seq is not None:
            seq_len = target_seq.size(1)
        else:
            seq_len = self.max_seq_len
        
        # Initialiser la première entrée (peut être zéro ou le premier point)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=device)
        
        outputs = []
        
        for t in range(seq_len):
            # Forward pass à travers le LSTM
            lstm_out, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
            
            # Prédire le prochain point
            output = self.fc_out(lstm_out)
            outputs.append(output)
            
            # Teacher forcing: utiliser la vraie valeur ou la prédiction
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t:t+1, :]
            else:
                decoder_input = output
        
        # Concaténer toutes les sorties
        output = torch.cat(outputs, dim=1)
        
        return output


class VAERNN(nn.Module):
    """
    Auto-Encodeur Variationnel Récurrent complet pour la génération
    de séquences d'écriture manuscrite.
    """
    
    def __init__(
        self,
        input_dim=2,
        hidden_dim=256,
        num_layers=2,
        latent_dim=32,
        max_seq_len=200,
        dropout=0.1
    ):
        """
        Args:
            input_dim: Dimension des entrées (2 pour x, y ou 3 pour x, y, pen_state)
            hidden_dim: Dimension de l'état caché LSTM
            num_layers: Nombre de couches LSTM empilées
            latent_dim: Dimension de l'espace latent
            max_seq_len: Longueur maximale des séquences
            dropout: Taux de dropout
        """
        super(VAERNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        
        # Encodeur
        self.encoder = EncoderRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        # Décodeur
        self.decoder = DecoderRNN(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=input_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
    
    def reparameterize(self, mu, logvar):
        """
        Rééchantillonnage de l'espace latent (trick de reparamétrisation).
        
        Args:
            mu: Moyenne de la distribution (batch_size, latent_dim)
            logvar: Log-variance de la distribution (batch_size, latent_dim)
        
        Returns:
            z: Vecteur latent échantillonné (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x):
        """
        Encode une séquence en représentation latente.
        
        Args:
            x: Séquence d'entrée (batch_size, seq_len, input_dim)
        
        Returns:
            mu: Moyenne de la distribution latente
            logvar: Log-variance de la distribution latente
            z: Vecteur latent échantillonné
        """
        mu, logvar, _ = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z
    
    def decode(self, z, target_seq=None, teacher_forcing_ratio=0.5):
        """
        Décode un vecteur latent en séquence.
        
        Args:
            z: Vecteur latent (batch_size, latent_dim)
            target_seq: Séquence cible pour teacher forcing (optionnel)
            teacher_forcing_ratio: Probabilité d'utiliser teacher forcing
        
        Returns:
            output: Séquence générée (batch_size, seq_len, output_dim)
        """
        return self.decoder(z, target_seq, teacher_forcing_ratio)
    
    def forward(self, x, teacher_forcing_ratio=0.5):
        """
        Forward pass complet: encode puis decode.
        
        Args:
            x: Séquence d'entrée (batch_size, seq_len, input_dim)
            teacher_forcing_ratio: Probabilité d'utiliser teacher forcing
        
        Returns:
            recon_x: Séquence reconstruite (batch_size, seq_len, input_dim)
            mu: Moyenne de la distribution latente
            logvar: Log-variance de la distribution latente
            z: Vecteur latent échantillonné
        """
        # Encoder
        mu, logvar, z = self.encode(x)
        
        # Decoder
        recon_x = self.decode(z, target_seq=x, teacher_forcing_ratio=teacher_forcing_ratio)
        
        return recon_x, mu, logvar, z
    
    def generate(self, z=None, num_samples=1, seq_len=None):
        """
        Génère de nouvelles séquences à partir de l'espace latent.
        
        Args:
            z: Vecteur(s) latent(s) (num_samples, latent_dim). Si None, échantillonne aléatoirement.
            num_samples: Nombre de séquences à générer si z est None
            seq_len: Longueur de la séquence à générer (par défaut max_seq_len)
        
        Returns:
            generated_seqs: Séquences générées (num_samples, seq_len, input_dim)
        """
        self.eval()
        
        if z is None:
            # Échantillonner aléatoirement dans l'espace latent
            z = torch.randn(num_samples, self.latent_dim)
        
        if seq_len is None:
            seq_len = self.max_seq_len
        
        # Générer la séquence
        with torch.no_grad():
            generated = self.decode(z, target_seq=None, teacher_forcing_ratio=0.0)
        
        return generated
    
    def interpolate(self, z1, z2, num_steps=10):
        """
        Interpole entre deux vecteurs latents pour générer des séquences intermédiaires.
        
        Args:
            z1: Premier vecteur latent (batch_size, latent_dim)
            z2: Deuxième vecteur latent (batch_size, latent_dim)
            num_steps: Nombre de points d'interpolation
        
        Returns:
            interpolated_seqs: Séquences interpolées (num_steps, seq_len, input_dim)
        """
        self.eval()
        
        # Interpolation linéaire dans l'espace latent
        alphas = torch.linspace(0, 1, num_steps).unsqueeze(1).to(z1.device)
        
        interpolated_z = []
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            interpolated_z.append(z_interp)
        
        interpolated_z = torch.cat(interpolated_z, dim=0)
        
        # Générer les séquences
        with torch.no_grad():
            interpolated_seqs = self.generate(z=interpolated_z)
        
        return interpolated_seqs

