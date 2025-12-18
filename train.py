"""
Script d'entraînement pour le VAE-RNN.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.vae_rnn import VAERNN
from utils.data_loader import get_data_loader, create_dummy_data
from utils.losses import vae_loss


def train_epoch(model, dataloader, optimizer, device, teacher_forcing_ratio=0.5, beta=1.0):
    """
    Entraîne le modèle pour une époque.
    
    Args:
        model: Modèle VAE-RNN
        dataloader: DataLoader pour les données d'entraînement
        optimizer: Optimiseur
        device: Device (cuda ou cpu)
        teacher_forcing_ratio: Probabilité d'utiliser teacher forcing
        beta: Poids de la divergence KL
    
    Returns:
        Époque moyenne des pertes
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch = batch.to(device)
        
        # Forward pass
        recon_batch, mu, logvar, z = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # Calculer la perte
        loss, recon_loss, kl_loss = vae_loss(recon_batch, batch, mu, logvar, beta=beta)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping pour stabiliser l'entraînement
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumuler les pertes
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    # Moyennes
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def validate(model, dataloader, device, beta=1.0):
    """
    Valide le modèle.
    
    Args:
        model: Modèle VAE-RNN
        dataloader: DataLoader pour les données de validation
        device: Device (cuda ou cpu)
        beta: Poids de la divergence KL
    
    Returns:
        Moyennes des pertes de validation
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = batch.to(device)
            
            # Forward pass (sans teacher forcing en validation)
            recon_batch, mu, logvar, z = model(batch, teacher_forcing_ratio=0.0)
            
            # Calculer la perte
            loss, recon_loss, kl_loss = vae_loss(recon_batch, batch, mu, logvar, beta=beta)
            
            # Accumuler les pertes
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    # Moyennes
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """
    Sauvegarde un checkpoint du modèle.
    
    Args:
        model: Modèle à sauvegarder
        optimizer: Optimiseur
        epoch: Numéro de l'époque
        loss: Perte actuelle
        checkpoint_dir: Répertoire de sauvegarde
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint sauvegardé: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Charge un checkpoint.
    
    Args:
        model: Modèle à charger
        optimizer: Optimiseur
        checkpoint_path: Chemin vers le checkpoint
    
    Returns:
        Époque de départ
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint chargé: {checkpoint_path}, époque {epoch}")
    return epoch


def plot_training_history(train_losses, val_losses, train_recon_losses, train_kl_losses, save_path):
    """
    Trace l'historique d'entraînement.
    
    Args:
        train_losses: Liste des pertes d'entraînement
        val_losses: Liste des pertes de validation
        train_recon_losses: Liste des pertes de reconstruction
        train_kl_losses: Liste des pertes KL
        save_path: Chemin de sauvegarde du graphique
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Perte totale
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss totale')
    plt.legend()
    plt.grid(True)
    
    # Perte de reconstruction
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_recon_losses, 'b-', label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Perte de reconstruction')
    plt.legend()
    plt.grid(True)
    
    # Perte KL
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_kl_losses, 'g-', label='KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Perte de divergence KL')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Graphique sauvegardé: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Entraînement du VAE-RNN pour la génération de séquences d\'écriture manuscrite')
    
    # Arguments pour les données
    parser.add_argument('--data_path', type=str, default='data/dummy_handwriting.npy',
                        help='Chemin vers les données d\'entraînement')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='Chemin vers les données de validation (optionnel)')
    parser.add_argument('--create_dummy_data', action='store_true',
                        help='Créer des données factices si les données n\'existent pas')
    
    # Arguments pour le modèle
    parser.add_argument('--input_dim', type=int, default=2,
                        help='Dimension des entrées (2 pour x,y ou 3 pour x,y,pen_state)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Dimension de l\'état caché LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Nombre de couches LSTM')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension de l\'espace latent')
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='Longueur maximale des séquences')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Taux de dropout')
    
    # Arguments pour l'entraînement
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Taille du batch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Nombre d\'époques')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Taux d\'apprentissage')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Poids de la divergence KL (beta-VAE)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='Probabilité d\'utiliser teacher forcing')
    
    # Arguments pour la sauvegarde
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Répertoire pour sauvegarder les checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Chemin vers un checkpoint à reprendre')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Sauvegarder un checkpoint tous les N époques')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilisé: {device}")
    
    # Créer des données factices si nécessaire
    data_path_to_use = args.data_path
    if args.create_dummy_data or not os.path.exists(args.data_path):
        print("Création de données factices...")
        create_dummy_data(num_samples=500, output_path=args.data_path)
        # Mettre à jour le chemin si l'extension a été changée
        if args.data_path.endswith('.npy'):
            pkl_path = args.data_path.replace('.npy', '.pkl')
            if os.path.exists(pkl_path):
                data_path_to_use = pkl_path
    
    # Charger les données
    print("Chargement des données...")
    train_loader, train_dataset = get_data_loader(
        data_path_to_use,
        batch_size=args.batch_size,
        shuffle=True,
        normalize=True,
        max_seq_len=args.max_seq_len
    )
    
    if args.val_data_path and os.path.exists(args.val_data_path):
        val_loader, _ = get_data_loader(
            args.val_data_path,
            batch_size=args.batch_size,
            shuffle=False,
            normalize=True,
            max_seq_len=args.max_seq_len
        )
    else:
        val_loader = None
    
    # Créer le modèle
    model = VAERNN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    ).to(device)
    
    print(f"Modèle créé: {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Optimiseur
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Reprendre depuis un checkpoint si spécifié
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)
    
    # Historique d'entraînement
    train_losses = []
    val_losses = []
    train_recon_losses = []
    train_kl_losses = []
    
    # Boucle d'entraînement
    print("Début de l'entraînement...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoque {epoch + 1}/{args.epochs}")
        
        # Entraînement
        train_loss, train_recon_loss, train_kl_loss = train_epoch(
            model, train_loader, optimizer, device,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            beta=args.beta
        )
        
        train_losses.append(train_loss)
        train_recon_losses.append(train_recon_loss)
        train_kl_losses.append(train_kl_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f}")
        
        # Validation
        if val_loader is not None:
            val_loss, val_recon_loss, val_kl_loss = validate(
                model, val_loader, device, beta=args.beta
            )
            val_losses.append(val_loss)
            print(f"Val Loss: {val_loss:.4f}, Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f}")
        else:
            val_losses.append(train_loss)  # Utiliser train_loss comme placeholder
        
        # Sauvegarder le checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, train_loss, args.checkpoint_dir)
    
    # Sauvegarder le modèle final
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"\nModèle final sauvegardé: {final_checkpoint_path}")
    
    # Tracer l'historique d'entraînement
    plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
    plot_training_history(train_losses, val_losses, train_recon_losses, train_kl_losses, plot_path)
    
    print("Entraînement terminé!")


if __name__ == '__main__':
    main()

