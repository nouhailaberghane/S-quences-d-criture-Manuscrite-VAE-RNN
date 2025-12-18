"""
Script de génération de séquences d'écriture manuscrite à partir du modèle VAE-RNN.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from models.vae_rnn import VAERNN
from utils.data_loader import HandwritingDataset


def generate_samples(model, num_samples=10, device='cuda', latent_dim=32):
    """
    Génère de nouvelles séquences à partir de l'espace latent.
    
    Args:
        model: Modèle VAE-RNN entraîné
        num_samples: Nombre de séquences à générer
        device: Device (cuda ou cpu)
        latent_dim: Dimension de l'espace latent
    
    Returns:
        generated_seqs: Séquences générées (num_samples, seq_len, input_dim)
    """
    model.eval()
    
    # Échantillonner aléatoirement dans l'espace latent
    z = torch.randn(num_samples, latent_dim).to(device)
    
    # Générer les séquences
    with torch.no_grad():
        generated_seqs = model.generate(z=z)
    
    return generated_seqs.cpu().numpy()


def interpolate_between_samples(model, dataset, idx1, idx2, num_steps=10, device='cuda'):
    """
    Interpole entre deux séquences réelles pour générer des séquences intermédiaires.
    
    Args:
        model: Modèle VAE-RNN entraîné
        dataset: Dataset contenant les séquences originales
        idx1: Index de la première séquence
        idx2: Index de la deuxième séquence
        num_steps: Nombre de points d'interpolation
        device: Device (cuda ou cpu)
    
    Returns:
        interpolated_seqs: Séquences interpolées (num_steps, seq_len, input_dim)
    """
    model.eval()
    
    # Encoder les deux séquences
    seq1 = dataset[idx1].unsqueeze(0).to(device)
    seq2 = dataset[idx2].unsqueeze(0).to(device)
    
    with torch.no_grad():
        mu1, logvar1, z1 = model.encode(seq1)
        mu2, logvar2, z2 = model.encode(seq2)
    
    # Interpoler dans l'espace latent
    interpolated_seqs = model.interpolate(z1, z2, num_steps=num_steps)
    
    return interpolated_seqs.cpu().numpy()


def plot_sequences(sequences, title="Séquences générées", save_path=None, denormalize_fn=None):
    """
    Trace les séquences d'écriture manuscrite.
    
    Args:
        sequences: Array de séquences (num_samples, seq_len, 2)
        title: Titre du graphique
        save_path: Chemin de sauvegarde (optionnel)
        denormalize_fn: Fonction de dénormalisation (optionnel)
    """
    num_samples = len(sequences)
    
    # Calculer la grille de sous-graphiques
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    for i, seq in enumerate(sequences):
        if i >= len(axes):
            break
        
        ax = axes[i]
        
        # Dénormaliser si nécessaire
        if denormalize_fn is not None:
            seq = denormalize_fn(seq)
        
        # Tracer la trajectoire
        x = seq[:, 0]
        y = seq[:, 1]
        
        ax.plot(x, y, 'b-', linewidth=1.5, alpha=0.7)
        ax.scatter(x[0], y[0], color='green', s=50, marker='o', label='Début', zorder=5)
        ax.scatter(x[-1], y[-1], color='red', s=50, marker='s', label='Fin', zorder=5)
        
        ax.set_title(f'Séquence {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        if i == 0:
            ax.legend()
    
    # Masquer les axes inutilisés
    for i in range(len(sequences), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graphique sauvegardé: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_interpolation(interpolated_seqs, title="Interpolation dans l'espace latent", save_path=None, denormalize_fn=None):
    """
    Trace une séquence d'interpolation.
    
    Args:
        interpolated_seqs: Séquences interpolées (num_steps, seq_len, 2)
        title: Titre du graphique
        save_path: Chemin de sauvegarde (optionnel)
        denormalize_fn: Fonction de dénormalisation (optionnel)
    """
    num_steps = len(interpolated_seqs)
    
    fig, axes = plt.subplots(2, (num_steps + 1) // 2, figsize=(20, 8))
    if num_steps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, seq in enumerate(interpolated_seqs):
        ax = axes[i]
        
        # Dénormaliser si nécessaire
        if denormalize_fn is not None:
            seq = denormalize_fn(seq)
        
        # Tracer la trajectoire
        x = seq[:, 0]
        y = seq[:, 1]
        
        ax.plot(x, y, 'b-', linewidth=1.5, alpha=0.7)
        ax.scatter(x[0], y[0], color='green', s=50, marker='o', zorder=5)
        ax.scatter(x[-1], y[-1], color='red', s=50, marker='s', zorder=5)
        
        ax.set_title(f'Étape {i+1}/{num_steps}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # Masquer les axes inutilisés
    for i in range(num_steps, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graphique sauvegardé: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Génération de séquences d\'écriture manuscrite avec VAE-RNN')
    
    # Arguments pour le modèle
    parser.add_argument('--model_path', type=str, required=True,
                        help='Chemin vers le modèle entraîné (.pth)')
    parser.add_argument('--input_dim', type=int, default=2,
                        help='Dimension des entrées')
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
    
    # Arguments pour la génération
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Nombre de séquences à générer')
    parser.add_argument('--interpolate', action='store_true',
                        help='Effectuer une interpolation entre deux séquences')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Chemin vers les données (nécessaire pour interpolation)')
    parser.add_argument('--idx1', type=int, default=0,
                        help='Index de la première séquence pour interpolation')
    parser.add_argument('--idx2', type=int, default=1,
                        help='Index de la deuxième séquence pour interpolation')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Nombre de points d\'interpolation')
    
    # Arguments pour la sauvegarde
    parser.add_argument('--output_dir', type=str, default='generated',
                        help='Répertoire de sauvegarde des résultats')
    parser.add_argument('--save_plots', action='store_true',
                        help='Sauvegarder les graphiques')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilisé: {device}")
    
    # Créer le modèle
    model = VAERNN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    ).to(device)
    
    # Charger les poids du modèle
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Modèle chargé: {args.model_path}")
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fonction de dénormalisation (si nécessaire)
    denormalize_fn = None
    if args.data_path and os.path.exists(args.data_path):
        dataset = HandwritingDataset(args.data_path, normalize=True, max_seq_len=args.max_seq_len)
        denormalize_fn = dataset.denormalize
    
    if args.interpolate:
        # Interpolation entre deux séquences
        if args.data_path is None or not os.path.exists(args.data_path):
            print("Erreur: --data_path est requis pour l'interpolation")
            return
        
        print(f"Interpolation entre les séquences {args.idx1} et {args.idx2}...")
        interpolated_seqs = interpolate_between_samples(
            model, dataset, args.idx1, args.idx2, args.num_steps, device
        )
        
        # Tracer l'interpolation
        plot_path = os.path.join(args.output_dir, 'interpolation.png') if args.save_plots else None
        plot_interpolation(interpolated_seqs, save_path=plot_path, denormalize_fn=denormalize_fn)
        
        # Sauvegarder les séquences
        np.save(os.path.join(args.output_dir, 'interpolated_sequences.npy'), interpolated_seqs)
        print(f"Séquences interpolées sauvegardées: {args.output_dir}/interpolated_sequences.npy")
    
    else:
        # Génération de nouvelles séquences
        print(f"Génération de {args.num_samples} séquences...")
        generated_seqs = generate_samples(model, args.num_samples, device, args.latent_dim)
        
        # Tracer les séquences générées
        plot_path = os.path.join(args.output_dir, 'generated_sequences.png') if args.save_plots else None
        plot_sequences(generated_seqs, save_path=plot_path, denormalize_fn=denormalize_fn)
        
        # Sauvegarder les séquences
        np.save(os.path.join(args.output_dir, 'generated_sequences.npy'), generated_seqs)
        print(f"Séquences générées sauvegardées: {args.output_dir}/generated_sequences.npy")
    
    print("Génération terminée!")


if __name__ == '__main__':
    main()

