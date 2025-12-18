"""
Utilitaires pour le chargement et le prétraitement des données
de séquences d'écriture manuscrite.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pickle


class HandwritingDataset(Dataset):
    """
    Dataset pour les séquences d'écriture manuscrite.
    
    Format attendu: chaque séquence est un array numpy de forme (seq_len, 2) ou (seq_len, 3)
    où les colonnes représentent (x, y) ou (x, y, pen_state).
    """
    
    def __init__(self, data_path, normalize=True, max_seq_len=None, pad_value=0.0):
        """
        Args:
            data_path: Chemin vers le fichier de données (numpy array ou pickle)
            normalize: Si True, normalise les données
            max_seq_len: Longueur maximale des séquences (padding/truncation)
            pad_value: Valeur pour le padding
        """
        self.pad_value = pad_value
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        
        # Charger les données
        if data_path.endswith('.npy'):
            self.data = np.load(data_path, allow_pickle=True)
        elif data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            raise ValueError(f"Format de fichier non supporté: {data_path}")
        
        # Convertir en liste si nécessaire
        if isinstance(self.data, np.ndarray):
            self.data = [self.data[i] for i in range(len(self.data))]
        
        # Normaliser les données
        if normalize:
            self._normalize_data()
        
        # Appliquer padding/truncation si nécessaire
        if max_seq_len is not None:
            self._pad_and_truncate()
    
    def _normalize_data(self):
        """Normalise les coordonnées x, y entre -1 et 1."""
        all_coords = []
        for seq in self.data:
            all_coords.append(seq[:, :2])  # Seulement x, y
        
        all_coords = np.concatenate(all_coords, axis=0)
        
        # Calculer les statistiques globales
        self.x_min, self.x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        self.y_min, self.y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        
        # Normaliser chaque séquence
        normalized_data = []
        for seq in self.data:
            normalized_seq = seq.copy()
            # Normaliser x, y entre -1 et 1
            normalized_seq[:, 0] = 2 * (seq[:, 0] - self.x_min) / (self.x_max - self.x_min) - 1
            normalized_seq[:, 1] = 2 * (seq[:, 1] - self.y_min) / (self.y_max - self.y_min) - 1
            normalized_data.append(normalized_seq)
        
        self.data = normalized_data
    
    def _pad_and_truncate(self):
        """Applique padding ou troncature pour uniformiser la longueur des séquences."""
        processed_data = []
        
        for seq in self.data:
            seq_len = len(seq)
            
            if seq_len > self.max_seq_len:
                # Tronquer
                processed_seq = seq[:self.max_seq_len]
            elif seq_len < self.max_seq_len:
                # Padding
                pad_length = self.max_seq_len - seq_len
                pad = np.full((pad_length, seq.shape[1]), self.pad_value)
                processed_seq = np.vstack([seq, pad])
            else:
                processed_seq = seq
            
            processed_data.append(processed_seq)
        
        self.data = processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.FloatTensor(seq)
    
    def denormalize(self, normalized_seq):
        """
        Dénormalise une séquence normalisée.
        
        Args:
            normalized_seq: Séquence normalisée (seq_len, 2) ou (seq_len, 3)
        
        Returns:
            Séquence dénormalisée
        """
        if not self.normalize:
            return normalized_seq
        
        denormalized = normalized_seq.copy()
        if isinstance(denormalized, torch.Tensor):
            denormalized = denormalized.cpu().numpy()
        
        # Dénormaliser x, y
        denormalized[:, 0] = (denormalized[:, 0] + 1) / 2 * (self.x_max - self.x_min) + self.x_min
        denormalized[:, 1] = (denormalized[:, 1] + 1) / 2 * (self.y_max - self.y_min) + self.y_min
        
        return denormalized


def create_dummy_data(num_samples=100, output_path='data/dummy_handwriting.npy'):
    """
    Crée des données factices pour tester le modèle.
    Génère des trajectoires sinusoïdales variées.
    
    Args:
        num_samples: Nombre de séquences à générer
        output_path: Chemin de sauvegarde
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sequences = []
    
    for i in range(num_samples):
        # Générer une trajectoire sinusoïdale variée
        seq_len = np.random.randint(50, 200)
        t = np.linspace(0, 4 * np.pi, seq_len)
        
        # Paramètres aléatoires pour varier les trajectoires
        amplitude_x = np.random.uniform(0.5, 2.0)
        amplitude_y = np.random.uniform(0.5, 2.0)
        frequency_x = np.random.uniform(0.5, 2.0)
        frequency_y = np.random.uniform(0.5, 2.0)
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)
        
        x = amplitude_x * np.sin(frequency_x * t + phase_x)
        y = amplitude_y * np.cos(frequency_y * t + phase_y)
        
        # Ajouter un peu de bruit
        x += np.random.normal(0, 0.1, seq_len)
        y += np.random.normal(0, 0.1, seq_len)
        
        # Créer la séquence (x, y)
        seq = np.column_stack([x, y])
        sequences.append(seq)
    
    # Sauvegarder en format pickle pour gérer les longueurs variables
    if output_path.endswith('.npy'):
        # Si l'extension est .npy, sauvegarder quand même en pickle mais avec extension .pkl
        output_path = output_path.replace('.npy', '.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(sequences, f)
    print(f"Données factices créées: {num_samples} séquences sauvegardées dans {output_path}")
    
    return sequences


def get_data_loader(data_path, batch_size=32, shuffle=True, **dataset_kwargs):
    """
    Crée un DataLoader PyTorch pour les données d'écriture manuscrite.
    
    Args:
        data_path: Chemin vers les données
        batch_size: Taille du batch
        shuffle: Si True, mélange les données
        **dataset_kwargs: Arguments additionnels pour HandwritingDataset
    
    Returns:
        DataLoader
    """
    dataset = HandwritingDataset(data_path, **dataset_kwargs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader, dataset

