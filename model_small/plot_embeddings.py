import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from pathlib import Path

# Configuration
EMBEDDINGS_DIR = r"C:\MIX\ASU\SEM_4\DSE 570\project_soft\model_small\embeddings"
USE_TSNE = False  # Set to True for t-SNE (slower but often better visualization)
PERPLEXITY = 30  # For t-SNE (adjust if needed)

def load_embeddings(filepath):
    """Load numpy embeddings file."""
    try:
        data = np.load(filepath)
        print(f"Loaded {filepath}: shape {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_labels(filepath):
    """Load labels file."""
    try:
        data = np.load(filepath)
        print(f"Loaded {filepath}: shape {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def reduce_dimensionality(embeddings, method='pca'):
    """Reduce embeddings to 2D for visualization."""
    if len(embeddings) == 0:
        return None
    
    print(f"Reducing {len(embeddings)} embeddings to 2D using {method.upper()}...")
    
    try:
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(embeddings)
            print(f"PCA explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
        else:  # t-SNE
            reducer = TSNE(n_components=2, perplexity=min(PERPLEXITY, len(embeddings)-1), random_state=42)
            reduced = reducer.fit_transform(embeddings)
        return reduced
    except Exception as e:
        print(f"Error reducing dimensionality: {e}")
        return None

def plot_embeddings(embeddings_2d, labels=None, title="Embeddings Visualization", save_path=None):
    """Plot 2D embeddings."""
    if embeddings_2d is None or len(embeddings_2d) == 0:
        print(f"Skipping plot for {title}: No data")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if labels is not None:
        # Flatten labels if needed
        if len(labels.shape) > 1:
            labels = labels.flatten()
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[colors[i]], label=f"Class {label}", alpha=0.6, s=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                  alpha=0.6, s=20, c='blue')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def main():
    print("=" * 70)
    print("EMBEDDING VISUALIZATION TOOL")
    print("=" * 70)
    print(f"Embeddings directory: {EMBEDDINGS_DIR}")
    print(f"Dimensionality reduction method: {'t-SNE' if USE_TSNE else 'PCA'}")
    print("=" * 70 + "\n")
    
    # Define embedding files to process
    embedding_files = {
        'pkl_labeled_embeddings.npy': 'pkl_labeled_labels.npy',
        'test_embeddings.npy': 'test_labels.npy',
        'train_embeddings.npy': 'train_labels.npy',
        'val_embeddings.npy': 'val_labels.npy',
        'unlabeled_embeddings.npy': None,
    }
    
    method = 'tsne' if USE_TSNE else 'pca'
    
    # Create output directory for plots
    output_dir = os.path.join(EMBEDDINGS_DIR, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each embedding file
    for emb_file, label_file in embedding_files.items():
        emb_path = os.path.join(EMBEDDINGS_DIR, emb_file)
        
        if not os.path.exists(emb_path):
            print(f"⚠ File not found: {emb_file}\n")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {emb_file}")
        print('='*70)
        
        # Load embeddings
        embeddings = load_embeddings(emb_path)
        if embeddings is None:
            continue
        
        # Load labels if available
        labels = None
        if label_file:
            label_path = os.path.join(EMBEDDINGS_DIR, label_file)
            if os.path.exists(label_path):
                labels = load_labels(label_path)
        
        # Reduce dimensionality
        embeddings_2d = reduce_dimensionality(embeddings, method)
        if embeddings_2d is None:
            continue
        
        # Plot
        title = f"{emb_file.replace('_embeddings.npy', '').title()} Embeddings"
        save_path = os.path.join(output_dir, f"{emb_file.replace('.npy', '')}_{method}.png")
        plot_embeddings(embeddings_2d, labels, title, save_path)
    
    print(f"\n{'='*70}")
    print(f"All plots saved to: {output_dir}")
    print('='*70)

if __name__ == "__main__":
    main()
