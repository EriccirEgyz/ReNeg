#!/usr/bin/env python3
"""
Analyze diversity of hard negatives using embeddings and clustering.

This script:
1) Loads training data with rejected_response (hardnegs) fields.
2) Encodes all hardnegs using a SentenceTransformer model.
3) Computes diversity metrics (variance-based).
4) Performs K-means clustering and analyzes cluster distribution.
5) Saves results to JSON and text report.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--device_count", type=int, required=True, help="Number of GPU devices to use")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the embedding model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (used for caching)")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset (used for caching)")

    # Processing arguments
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for encoding")

    # Clustering arguments
    parser.add_argument("--n_clusters", type=int, required=True, help="Number of clusters for K-means")
    parser.add_argument("--random_seed", type=int, required=True, help="Random seed for reproducibility")

    # Cache arguments
    parser.add_argument("--save_encodings", type=int, choices=[0, 1], required=True, help="Save embeddings to cache (1) or not (0)")

    return parser.parse_args()

def load_or_compute_encodings(
    model: SentenceTransformer,
    texts: List[str],
    cache_path: str,
    batch_size: int,
    device: str,
    save_encodings: bool
) -> np.ndarray:
    """
    Load cached embeddings or compute new ones.

    Args:
        model: SentenceTransformer model
        texts: List of texts to encode
        cache_path: Path to cache file
        batch_size: Batch size for encoding
        device: Device to use for encoding
        save_encodings: Whether to save computed embeddings

    Returns:
        Embeddings array
    """
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        prompt="",
        batch_size=batch_size,
        show_progress_bar=True,
        device=device
    ).astype(np.float32)

    if save_encodings:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"Saved embeddings to {cache_path}")

    return embeddings

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors along axis 1."""
    norms = np.linalg.norm(vectors, ord=2, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def compute_clustering_metrics(
    embeddings: np.ndarray,
    n_clusters: int,
    random_seed: int
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Perform K-means clustering and compute cluster metrics.

    Args:
        embeddings: Hardneg embeddings
        n_clusters: Number of clusters
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (metrics, labels, cluster_centers)
    """
    print(f"Performing K-means clustering with k={n_clusters}...")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        n_init=5,
        random_state=random_seed
    )
    labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    probs = counts / counts.sum()
    cluster_entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    effective_num_clusters = float(np.exp(cluster_entropy))

    metrics = {
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
        'cluster_entropy': cluster_entropy,
        'effective_num_clusters': effective_num_clusters,
        'min_cluster_size': int(np.min(counts)),
        'max_cluster_size': int(np.max(counts)),
        'avg_cluster_size': float(np.mean(counts))
    }

    return metrics, labels, cluster_centers

def save_results(
    global_metrics: Dict,
    cluster_metrics: Dict,
    output_dir: str
) -> None:
    """
    Save analysis results to files.

    Args:
        global_metrics: Global diversity metrics
        cluster_metrics: Clustering metrics
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'global_diversity_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(global_metrics, f, indent=2, ensure_ascii=False)

    with open(output_path / 'cluster_diversity_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(cluster_metrics, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - global_metrics.json")
    print(f"  - cluster_metrics.json")

def main():
    """Main function."""
    args = parse_arguments()

    print("=" * 60)
    print("Hardneg Diversity Analysis")
    print("=" * 60)

    # [Step 1] Load data
    print("\n[Step 1] Loading data...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(data)} records")

    # [Step 2] Extract hardnegs
    print("\n[Step 2] Extracting hardnegs...")
    all_hardnegs = []
    query_count = 0
    for item in data:
        if 'rejected_response' in item and item['rejected_response']:
            query_count += 1
            all_hardnegs.extend(item['rejected_response'])

    print(f"Found {query_count} queries with hardnegs")
    print(f"Total hardnegs: {len(all_hardnegs)}")

    if len(all_hardnegs) == 0:
        print("\nError: No hardnegs found in dataset")
        return

    # [Step 3] Load model
    print(f"\n[Step 3] Loading model from {args.model_path}...")
    model = SentenceTransformer(args.model_path, trust_remote_code=True)
    device = [f"cuda:{i}" for i in range(args.device_count)] if args.device_count > 1 else "cuda:0"
    print(f"Model max sequence length: {model.get_max_seq_length()}")

    # [Step 4] Compute embeddings
    print("\n[Step 4] Computing hardneg embeddings...")
    cache_dir = f"../../cache/analyze_hardneg/{args.model_name}/{args.data_name}"
    cache_path = os.path.join(cache_dir, "hardneg_embeddings.npy")
    embeddings = load_or_compute_encodings(
        model, all_hardnegs, cache_path,
        args.batch_size, device, args.save_encodings == 1
    )

    # [Step 5] Compute global diversity metrics
    print("\n[Step 5] Computing global diversity metrics...")
    variance_per_dim = np.var(embeddings, axis=0)
    std_per_dim = np.std(embeddings, axis=0)

    global_metrics = {
        'mean_variance': float(np.mean(variance_per_dim)),
        'max_variance': float(np.max(variance_per_dim)),
        'min_variance': float(np.min(variance_per_dim))
    }

    # [Step 6] Perform clustering analysis
    print("\n[Step 6] Performing clustering analysis...")
    cluster_metrics, labels, centers = compute_clustering_metrics(
        embeddings, args.n_clusters, args.random_seed
    )

    # [Step 7] Save results
    print("\n[Step 7] Saving results...")
    output_dir = f"../../results/analyze_hardneg/{args.model_name}/{args.data_name}"
    save_results(global_metrics, cluster_metrics, output_dir)

    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()