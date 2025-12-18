"""
Fonctions de perte pour l'entraînement du VAE-RNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(recon_x, x, reduction='mean'):
    """
    Perte de reconstruction (MSE entre la séquence originale et reconstruite).
    
    Args:
        recon_x: Séquence reconstruite (batch_size, seq_len, input_dim)
        x: Séquence originale (batch_size, seq_len, input_dim)
        reduction: 'mean' ou 'sum'
    
    Returns:
        Perte de reconstruction
    """
    mse = F.mse_loss(recon_x, x, reduction='none')
    
    # Somme sur les dimensions spatiales, moyenne sur la séquence
    mse = mse.sum(dim=2).mean(dim=1)
    
    if reduction == 'mean':
        return mse.mean()
    elif reduction == 'sum':
        return mse.sum()
    else:
        return mse


def kl_divergence_loss(mu, logvar, reduction='mean'):
    """
    Perte de divergence KL pour régulariser l'espace latent.
    
    Args:
        mu: Moyenne de la distribution latente (batch_size, latent_dim)
        logvar: Log-variance de la distribution latente (batch_size, latent_dim)
        reduction: 'mean' ou 'sum'
    
    Returns:
        Perte de divergence KL
    """
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:
        return kl


def vae_loss(recon_x, x, mu, logvar, beta=1.0, reduction='mean'):
    """
    Perte totale du VAE: reconstruction + beta * KL divergence.
    
    Args:
        recon_x: Séquence reconstruite (batch_size, seq_len, input_dim)
        x: Séquence originale (batch_size, seq_len, input_dim)
        mu: Moyenne de la distribution latente (batch_size, latent_dim)
        logvar: Log-variance de la distribution latente (batch_size, latent_dim)
        beta: Poids de la divergence KL (pour beta-VAE)
        reduction: 'mean' ou 'sum'
    
    Returns:
        loss: Perte totale
        recon_loss: Perte de reconstruction
        kl_loss: Perte de divergence KL
    """
    recon_loss = reconstruction_loss(recon_x, x, reduction=reduction)
    kl_loss = kl_divergence_loss(mu, logvar, reduction=reduction)
    
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss


class VAELoss(nn.Module):
    """
    Module PyTorch pour la perte VAE (peut être utilisé comme fonction de perte).
    """
    
    def __init__(self, beta=1.0, reduction='mean'):
        """
        Args:
            beta: Poids de la divergence KL
            reduction: 'mean' ou 'sum'
        """
        super(VAELoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, recon_x, x, mu, logvar):
        """
        Args:
            recon_x: Séquence reconstruite
            x: Séquence originale
            mu: Moyenne de la distribution latente
            logvar: Log-variance de la distribution latente
        
        Returns:
            loss: Perte totale
            recon_loss: Perte de reconstruction
            kl_loss: Perte de divergence KL
        """
        return vae_loss(recon_x, x, mu, logvar, self.beta, self.reduction)

