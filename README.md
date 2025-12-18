# Génération de Séquences d'Écriture Manuscrite via Auto-Encodeur Variationnel Récurrent (VAE-RNN)

Ce projet implémente un Auto-Encodeur Variationnel Récurrent (VAE-RNN) pour la génération de séquences d'écriture manuscrite. L'architecture combine un encodeur LSTM, un espace latent structuré, et un décodeur LSTM pour générer des trajectoires d'écriture cohérentes et fluides.

## Architecture

Le modèle VAE-RNN est composé de trois composants principaux:

1. **Encodeur RNN/LSTM**: Transforme les séquences d'écriture manuscrite (trajectoires x, y) en représentation latente
2. **Espace latent**: Distribution gaussienne permettant le contrôle du style et l'interpolation entre différentes écritures
3. **Décodeur RNN/LSTM**: Génère de nouvelles séquences à partir de l'espace latent, en modélisant la dépendance temporelle pour assurer la cohérence

## Structure du Projet

```
.
├── models/
│   └── vae_rnn.py          # Architecture VAE-RNN (Encodeur, Décodeur, VAE complet)
├── utils/
│   ├── data_loader.py      # Chargement et prétraitement des données
│   └── losses.py           # Fonctions de perte (reconstruction + KL divergence)
├── train.py                # Script d'entraînement
├── generate.py             # Script de génération et interpolation
├── requirements.txt        # Dépendances Python
└── README.md              # Documentation
```

## Installation

1. Cloner ou télécharger le projet

2. Installer les dépendances:
```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Préparation des Données

Les données doivent être au format numpy array ou pickle, où chaque élément est une séquence de forme `(seq_len, 2)` ou `(seq_len, 3)` représentant les coordonnées (x, y) ou (x, y, pen_state).

### 2. Entraînement

Entraîner le modèle avec les paramètres par défaut:
```bash
python train.py --data_path data/handwriting_data.npy
```

Options d'entraînement principales:
- `--data_path`: Chemin vers les données d'entraînement
- `--batch_size`: Taille du batch (défaut: 32)
- `--epochs`: Nombre d'époques (défaut: 100)
- `--lr`: Taux d'apprentissage (défaut: 0.001)
- `--latent_dim`: Dimension de l'espace latent (défaut: 32)
- `--hidden_dim`: Dimension de l'état caché LSTM (défaut: 256)
- `--beta`: Poids de la divergence KL pour beta-VAE (défaut: 1.0)
- `--checkpoint_dir`: Répertoire pour sauvegarder les checkpoints

Exemple avec paramètres personnalisés:
```bash
python train.py \
    --data_path data/handwriting_data.npy \
    --batch_size 64 \
    --epochs 200 \
    --lr 0.0005 \
    --latent_dim 64 \
    --hidden_dim 512 \
    --beta 0.5 \
    --checkpoint_dir checkpoints
```

### 3. Génération de Séquences

Générer de nouvelles séquences à partir de l'espace latent:
```bash
python generate.py \
    --model_path checkpoints/final_model.pth \
    --num_samples 10 \
    --save_plots
```

### 4. Interpolation dans l'Espace Latent

Interpoler entre deux séquences pour générer des transitions fluides:
```bash
python generate.py \
    --model_path checkpoints/final_model.pth \
    --interpolate \
    --data_path data/handwriting_data.npy \
    --idx1 0 \
    --idx2 5 \
    --num_steps 10 \
    --save_plots
```

## Paramètres du Modèle

### Architecture

- **input_dim**: Dimension des entrées (2 pour x, y ou 3 pour x, y, pen_state)
- **hidden_dim**: Dimension de l'état caché LSTM (recommandé: 256-512)
- **num_layers**: Nombre de couches LSTM empilées (recommandé: 2-3)
- **latent_dim**: Dimension de l'espace latent (recommandé: 16-64)
- **max_seq_len**: Longueur maximale des séquences
- **dropout**: Taux de dropout pour la régularisation

### Fonction de Perte

La perte totale est composée de deux termes:
- **Perte de reconstruction**: MSE entre les séquences originales et reconstruites
- **Divergence KL**: Régularisation de l'espace latent pour suivre une distribution gaussienne standard

```
Loss = Reconstruction_Loss + β * KL_Loss
```

Le paramètre `β` contrôle le compromis entre la qualité de reconstruction et la régularisation de l'espace latent.

## Fonctionnalités

### Génération de Nouvelles Séquences

Le modèle peut générer de nouvelles trajectoires d'écriture en échantillonnant dans l'espace latent. Chaque point de l'espace latent correspond à un style d'écriture différent.

### Interpolation de Style

L'interpolation linéaire dans l'espace latent permet de créer des transitions fluides entre différents styles d'écriture, démontrant la structure continue de l'espace latent.

### Contrôle du Style

En manipulant le vecteur latent, on peut contrôler les caractéristiques des séquences générées (courbure, vitesse, taille, etc.).

## Technologies Utilisées

- **Python**: Langage de programmation principal
- **PyTorch**: Framework de deep learning pour l'implémentation du VAE-RNN
- **NumPy**: Manipulation des données numériques
- **Scikit-learn**: Utilitaires de prétraitement (normalisation)
- **Matplotlib**: Visualisation des séquences générées

## Notes Techniques

### Teacher Forcing

Le décodeur utilise le teacher forcing pendant l'entraînement pour accélérer la convergence. Le ratio de teacher forcing peut être ajusté via le paramètre `--teacher_forcing_ratio`.

### Normalisation des Données

Les données sont automatiquement normalisées entre -1 et 1 pour améliorer la stabilité de l'entraînement. La fonction de dénormalisation est disponible pour visualiser les séquences dans leur échelle originale.

### Gradient Clipping

Un gradient clipping est appliqué pendant l'entraînement pour éviter l'explosion des gradients, particulièrement important pour les RNN.

## Résultats Attendus

Après l'entraînement, le modèle devrait être capable de:
- Reconstruire fidèlement les séquences d'entrée
- Générer de nouvelles séquences variées et cohérentes
- Interpoler entre différents styles d'écriture
- Maintenir la cohérence temporelle des trajectoires générées

## Améliorations Possibles

- Utilisation de données réelles d'écriture manuscrite (IAM OnDB, CASIA, etc.)
- Ajout d'un mécanisme d'attention dans le décodeur
- Implémentation d'un VAE conditionnel pour contrôler le contenu généré
- Utilisation de GAN pour améliorer la qualité des séquences générées
- Extension à des séquences 3D (avec pression du stylo)

## Auteur

Projet développé pour la génération de séquences d'écriture manuscrite via Auto-Encodeur Variationnel Récurrent.

