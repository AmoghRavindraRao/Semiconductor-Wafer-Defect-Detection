"""
visualize_embeddings_interactive.py

Interactive 3D UMAP + Plotly visualization of SupCon embeddings with misclassification overlay.

Processes ALL embedding files in the embeddings folder:
  - test_embeddings.npy + test_labels.npy
  - train_embeddings.npy + train_labels.npy
  - val_embeddings.npy + val_labels.npy
  - pkl_labeled_embeddings.npy + pkl_labeled_labels.npy
  - unlabeled_embeddings.npy (skipped - no labels)

For each embedding set, generates:
  - 3D reduction: 2D UMAP (cosine metric) + 1D PCA 
  - Misclassification overlay
  - Centroid markers
  - Full 360° rotation capability

Output: embeddings/plots/{dataset}_interactive_3d.html

Usage:
    python visualize_embeddings_interactive.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from umap import UMAP
from sklearn.decomposition import PCA

from utils import IDX_TO_CLASS, NUM_CLASSES, PATHS

# =============================================================================
# SETUP
# =============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

EMB_DIR = PATHS["emb"]
PLOT_DIR = EMB_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_centroids() -> np.ndarray:
    """Load centroids (shared across all datasets)."""
    centroids = np.load(EMB_DIR / "centroids.npy")
    assert centroids.shape[0] == NUM_CLASSES, f"Expected {NUM_CLASSES} centroids, got {centroids.shape[0]}"
    return centroids


def find_embedding_files() -> list:
    """
    Discover all embedding files in embeddings folder.
    
    Returns list of tuples: (dataset_name, emb_path, labels_path or None)
    """
    candidates = [
        ("test", "test_embeddings.npy", "test_labels.npy"),
        ("train", "train_embeddings.npy", "train_labels.npy"),
        ("val", "val_embeddings.npy", "val_labels.npy"),
        ("pkl_labeled", "pkl_labeled_embeddings.npy", "pkl_labeled_labels.npy"),
    ]
    
    found = []
    for name, emb_file, label_file in candidates:
        emb_path = EMB_DIR / emb_file
        label_path = EMB_DIR / label_file
        
        if emb_path.exists() and label_path.exists():
            found.append((name, emb_path, label_path))
            logger.info(f"Found: {name:15} ({emb_file})")
        else:
            logger.info(f"Skipped: {name:15} (missing {label_file if not label_path.exists() else emb_file})")
    
    return found


def load_data(emb_path: Path, labels_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels."""
    embeddings = np.load(emb_path)
    labels = np.load(labels_path)
    
    assert embeddings.shape[0] == labels.shape[0], "Mismatch between embeddings and labels"
    return embeddings, labels


# =============================================================================
# GET PREDICTIONS
# =============================================================================

def get_predictions(embeddings: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Predict class for each embedding as nearest centroid (cosine distance).
    
    Args:
        embeddings: (N, d) L2-normalized embeddings
        centroids: (9, d) centroid embeddings
        labels: (N,) true labels for accuracy reporting
        
    Returns:
        predictions: (N,) predicted class indices
    """
    logger.info("Computing predictions via nearest centroid...")
    
    # Normalize centroids to L2 norm (cosine metric)
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity: sim = emb @ centroid^T
    sim = embeddings_norm @ centroids_norm.T  # (N, 9)
    predictions = np.argmax(sim, axis=1)
    
    n_correct = np.sum(predictions == labels)
    acc = 100.0 * n_correct / len(predictions)
    logger.info(f"Accuracy (nearest centroid): {acc:.2f}% ({n_correct}/{len(predictions)})")
    
    return predictions


# =============================================================================
# UMAP REDUCTION
# =============================================================================

def reduce_embeddings(embeddings: np.ndarray) -> Tuple[np.ndarray, UMAP, np.ndarray]:
    """
    Reduce embeddings to 3D: 2D UMAP + 1D PCA.
    
    Args:
        embeddings: (N, d) L2-normalized embeddings
        
    Returns:
        reduced_3d: (N, 3) 3D coordinates [umap_x, umap_y, pca_z]
        reducer: fitted UMAP object
        pca: fitted PCA object for centroids
    """
    logger.info("Fitting UMAP reducer (2D)...")
    reducer = UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric='cosine',
        random_state=0,
        verbose=1,
    )
    umap_2d = reducer.fit_transform(embeddings)
    logger.info(f"UMAP reduced embeddings to shape {umap_2d.shape}")
    
    # Add z-axis from first PCA component
    logger.info("Fitting PCA for z-axis...")
    pca = PCA(n_components=1, random_state=0)
    pca_z = pca.fit_transform(embeddings)
    
    # Normalize z-axis to [-1, 1] for better visualization
    pca_z_norm = (pca_z - pca_z.min()) / (pca_z.max() - pca_z.min() + 1e-8) * 2 - 1
    
    reduced_3d = np.concatenate([umap_2d, pca_z_norm], axis=1)
    logger.info(f"Added z-axis (PCA): 3D shape {reduced_3d.shape}")
    
    return reduced_3d, reducer, pca


# =============================================================================
# CREATE INTERACTIVE PLOTLY FIGURE
# =============================================================================

def create_interactive_plot(
    reduced_coords: np.ndarray,
    test_labels: np.ndarray,
    predictions: np.ndarray,
    centroids_reduced: np.ndarray,
) -> go.Figure:
    """
    Create interactive 3D Plotly figure with all points, centroids, and misclassifications.
    
    Args:
        reduced_coords: (N, 3) 3D coordinates [umap_x, umap_y, pca_z]
        test_labels: (N,) true class labels
        predictions: (N,) predicted class labels
        centroids_reduced: (9, 3) 3D centroid coordinates
        
    Returns:
        fig: Plotly 3D scatter figure object
    """
    logger.info("Creating interactive 3D Plotly visualization...")
    
    # Identify misclassifications
    is_misclassified = test_labels != predictions
    n_misclass = np.sum(is_misclassified)
    logger.info(f"Misclassified points: {n_misclass} / {len(test_labels)} ({100*n_misclass/len(test_labels):.2f}%)")
    
    # Create figure
    fig = go.Figure()
    
    # Qualitative color palette (Set1 for 9 colors)
    colors = px.colors.qualitative.Set1  # 9 distinct colors
    
    # Add traces for each class
    for class_idx in range(NUM_CLASSES):
        class_name = IDX_TO_CLASS[class_idx]
        
        # Correct predictions for this class
        mask_correct = (test_labels == class_idx) & (~is_misclassified)
        x_correct = reduced_coords[mask_correct, 0]
        y_correct = reduced_coords[mask_correct, 1]
        z_correct = reduced_coords[mask_correct, 2]
        
        fig.add_trace(go.Scatter3d(
            x=x_correct,
            y=y_correct,
            z=z_correct,
            mode='markers',
            name=f"{class_name}",
            marker=dict(
                size=3,
                color=colors[class_idx % len(colors)],
                opacity=0.6,
                line=dict(width=0),
            ),
            text=[f"True: {class_name}<br>Pred: {IDX_TO_CLASS[predictions[i]]}<br>Correct" 
                  for i in np.where(mask_correct)[0]],
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        ))
        
        # Misclassified points from this class (color by true label, different marker)
        mask_misclass = (test_labels == class_idx) & is_misclassified
        x_misclass = reduced_coords[mask_misclass, 0]
        y_misclass = reduced_coords[mask_misclass, 1]
        z_misclass = reduced_coords[mask_misclass, 2]
        
        if len(x_misclass) > 0:
            fig.add_trace(go.Scatter3d(
                x=x_misclass,
                y=y_misclass,
                z=z_misclass,
                mode='markers',
                name=f"{class_name} (misclassified)",
                marker=dict(
                    size=5,
                    color=colors[class_idx % len(colors)],
                    opacity=0.8,
                    symbol='x',
                    line=dict(width=1, color='black'),
                ),
                text=[f"True: {class_name}<br>Pred: {IDX_TO_CLASS[predictions[i]]}<br>❌ MISCLASSIFIED" 
                      for i in np.where(mask_misclass)[0]],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            ))
    
    # Add centroids as a separate trace
    fig.add_trace(go.Scatter3d(
        x=centroids_reduced[:, 0],
        y=centroids_reduced[:, 1],
        z=centroids_reduced[:, 2],
        mode='markers+text',
        name="Centroids",
        marker=dict(
            size=8,
            color='black',
            symbol='diamond',
            line=dict(width=2, color='yellow'),
        ),
        text=[IDX_TO_CLASS[i] for i in range(NUM_CLASSES)],
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hovertemplate="<b>%{text} Centroid</b><extra></extra>",
        showlegend=True,
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': "SupCon Embedding Visualization (3D: UMAP 2D + PCA 1D)<br><sub>Rotate horizontally & vertically • Zoom • Toggle legend</sub>",
            'x': 0.5,
            'xanchor': 'center',
        },
        scene=dict(
            xaxis_title="UMAP Dim 1",
            yaxis_title="UMAP Dim 2",
            zaxis_title="PCA Dim 1",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            aspectmode='cube',
        ),
        width=1400,
        height=900,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
        ),
    )
    
    logger.info(f"3D Figure created with {NUM_CLASSES * 2} traces (correct + misclass per class) + centroids")
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Process all embedding files and generate 3D visualizations."""
    
    logger.info("\n" + "="*70)
    logger.info("FINDING EMBEDDING FILES")
    logger.info("="*70)
    
    # Load centroids (shared)
    centroids = load_centroids()
    logger.info(f"Loaded centroids: {centroids.shape}")
    
    # Find all embedding files
    embedding_files = find_embedding_files()
    if not embedding_files:
        logger.error("No embedding files found!")
        return
    
    logger.info(f"\nProcessing {len(embedding_files)} dataset(s)...\n")
    
    # Process each embedding file
    results = []
    for dataset_name, emb_path, labels_path in embedding_files:
        logger.info("="*70)
        logger.info(f"PROCESSING: {dataset_name.upper()}")
        logger.info("="*70)
        
        try:
            # Load data
            embeddings, labels = load_data(emb_path, labels_path)
            logger.info(f"Loaded: {embeddings.shape} embeddings, {labels.shape} labels")
            
            # Get predictions
            predictions = get_predictions(embeddings, centroids, labels)
            
            # Reduce to 3D
            reduced_coords, reducer, pca = reduce_embeddings(embeddings)
            
            # Reduce centroids to 3D
            centroids_umap = reducer.transform(centroids)
            centroids_pca = pca.transform(centroids)
            centroids_pca_norm = (centroids_pca - centroids_pca.min()) / (centroids_pca.max() - centroids_pca.min() + 1e-8) * 2 - 1
            centroids_reduced = np.concatenate([centroids_umap, centroids_pca_norm], axis=1)
            logger.info(f"Centroids reduced to {centroids_reduced.shape}")
            
            # Create interactive plot
            fig = create_interactive_plot(reduced_coords, labels, predictions, centroids_reduced)
            
            # Save as HTML
            output_path = PLOT_DIR / f"{dataset_name}_interactive_3d.html"
            fig.write_html(str(output_path), config=dict(responsive=True))
            
            n_misclass = np.sum(labels != predictions)
            acc = 100.0 * (len(labels) - n_misclass) / len(labels)
            
            logger.info(f"✓ Saved: {output_path.name}")
            logger.info(f"  Points: {len(embeddings):,} | Accuracy: {acc:.2f}% | Misclassified: {n_misclass}")
            
            results.append({
                'name': dataset_name,
                'n_points': len(embeddings),
                'accuracy': acc,
                'misclassified': n_misclass,
                'output': output_path.name,
            })
            
        except Exception as e:
            logger.error(f"✗ Failed to process {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Successfully generated {len(results)} visualization(s):\n")
    for r in results:
        logger.info(f"  {r['name']:12} | {r['n_points']:6,} pts | {r['accuracy']:5.2f}% acc | {r['misclassified']:5} misclass | {r['output']}")
    logger.info(f"\nAll files saved to: {PLOT_DIR}")
    logger.info("Open any .html file in a browser to explore!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
