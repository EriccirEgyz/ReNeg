#!/usr/bin/env python3
"""
Analyze hardness of hard negatives using query-hardneg and query-positive similarity.

This script:
1) Loads training data with rejected_response (hardnegs) and response (positive) fields.
2) Encodes queries, positives, and hardnegs using a SentenceTransformer model.
3) Computes similarity between each query and its hardnegs/positives.
4) Computes ratio: sim(query, hardneg) / sim(query, positive).
5) Generates histograms and saves statistics.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

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

def compute_cosine_similarity(query_emb: np.ndarray, hardneg_embs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and hardneg embeddings.

    Args:
        query_emb: Query embedding (1, dim)
        hardneg_embs: Hardneg embeddings (n, dim)

    Returns:
        Similarity scores (n,)
    """
    # Both should already be normalized
    return np.dot(hardneg_embs, query_emb.T).flatten()

def compute_hardness_metrics(
    query_embeddings: np.ndarray,
    hardneg_embeddings: np.ndarray,
    positive_embeddings: np.ndarray,
    query_hardneg_indices: List[Tuple[int, List[int]]]
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute hardness metrics based on query-hardneg and query-positive similarity.

    Args:
        query_embeddings: All query embeddings
        hardneg_embeddings: All hardneg embeddings (flattened)
        positive_embeddings: All positive embeddings
        query_hardneg_indices: List of (query_idx, [hardneg_indices]) tuples

    Returns:
        Tuple of (metrics_dict, all_similarities_qh, all_similarities_qp, all_ratios)
    """
    print("Computing query-hardneg and query-positive similarities...")

    all_similarities_qh = []  # query-hardneg similarities
    all_similarities_qp = []  # query-positive similarities
    all_ratios = []           # sim(q,h) / sim(q,p) ratios

    for query_idx, hardneg_indices in query_hardneg_indices:
        query_emb = query_embeddings[query_idx:query_idx+1]
        hardneg_embs = hardneg_embeddings[hardneg_indices]
        positive_emb = positive_embeddings[query_idx:query_idx+1]

        # Compute query-hardneg similarities
        similarities_qh = compute_cosine_similarity(query_emb, hardneg_embs)
        all_similarities_qh.extend(similarities_qh.tolist())

        # Compute query-positive similarity
        similarity_qp = compute_cosine_similarity(query_emb, positive_emb)[0]
        all_similarities_qp.append(similarity_qp)

        # Compute ratios: sim(q,h) / sim(q,p)
        ratios = similarities_qh / similarity_qp
        all_ratios.extend(ratios.tolist())

    all_similarities_qh = np.array(all_similarities_qh)
    all_similarities_qp = np.array(all_similarities_qp)
    all_ratios = np.array(all_ratios)

    metrics = {
        'num_queries': len(query_hardneg_indices),
        'total_hardnegs': len(all_similarities_qh),
        # Query-hardneg similarity metrics
        'qh_similarity_mean': float(np.mean(all_similarities_qh)),
        'qh_similarity_std': float(np.std(all_similarities_qh)),
        'qh_similarity_min': float(np.min(all_similarities_qh)),
        'qh_similarity_max': float(np.max(all_similarities_qh)),
        'qh_similarity_median': float(np.median(all_similarities_qh)),
        'qh_similarity_q25': float(np.percentile(all_similarities_qh, 25)),
        'qh_similarity_q75': float(np.percentile(all_similarities_qh, 75)),
        # Query-positive similarity metrics
        'qp_similarity_mean': float(np.mean(all_similarities_qp)),
        'qp_similarity_std': float(np.std(all_similarities_qp)),
        'qp_similarity_min': float(np.min(all_similarities_qp)),
        'qp_similarity_max': float(np.max(all_similarities_qp)),
        'qp_similarity_median': float(np.median(all_similarities_qp)),
        'qp_similarity_q25': float(np.percentile(all_similarities_qp, 25)),
        'qp_similarity_q75': float(np.percentile(all_similarities_qp, 75)),
        # Ratio metrics: sim(q,h) / sim(q,p)
        'ratio_mean': float(np.mean(all_ratios)),
        'ratio_std': float(np.std(all_ratios)),
        'ratio_min': float(np.min(all_ratios)),
        'ratio_max': float(np.max(all_ratios)),
        'ratio_median': float(np.median(all_ratios)),
        'ratio_q25': float(np.percentile(all_ratios, 25)),
        'ratio_q75': float(np.percentile(all_ratios, 75)),
    }

    return metrics, all_similarities_qh, all_similarities_qp, all_ratios

def plot_histogram(data: np.ndarray, output_path: str, title: str) -> None:
    """Generate and save histogram plot."""
    plt.hist(data, bins=50)
    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.savefig(output_path)
    plt.close()

def save_results(
    metrics: Dict,
    similarities_qh: np.ndarray,
    similarities_qp: np.ndarray,
    ratios: np.ndarray,
    output_dir: str
) -> None:
    """Save analysis results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    with open(output_path / 'hardness_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Generate histograms
    plot_histogram(
        similarities_qh,
        str(output_path / 'histogram_qh_similarity.png'),
        'Query-Hardneg Similarity Distribution'
    )
    plot_histogram(
        similarities_qp,
        str(output_path / 'histogram_qp_similarity.png'),
        'Query-Positive Similarity Distribution'
    )
    plot_histogram(
        ratios,
        str(output_path / 'histogram_ratio.png'),
        'Ratio Distribution: sim(query, hardneg) / sim(query, positive)'
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"  - hardness_metrics.json")
    print(f"  - histogram_qh_similarity.png")
    print(f"  - histogram_qp_similarity.png")
    print(f"  - histogram_ratio.png")

def main():
    """Main function."""
    args = parse_arguments()

    print("=" * 60)
    print("Hardneg Hardness Analysis")
    print("=" * 60)

    # [Step 1] Load data
    print("\n[Step 1] Loading data...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(data)} records")

    # [Step 2] Extract queries, positives, and hardnegs
    print("\n[Step 2] Extracting queries, positives, and hardnegs...")
    queries = []
    all_positives = []
    all_hardnegs = []
    query_hardneg_indices = []  # List of (query_idx, [hardneg_indices]) tuples

    hardneg_offset = 0
    query_idx = 0

    for item in data:
        query = item["query"]
        queries.append(query)

        all_positives.append(item["response"])

        if 'rejected_response' in item and item['rejected_response']:
            hardnegs = item['rejected_response']
            num_hardnegs = len(hardnegs)
            hardneg_indices = list(range(hardneg_offset, hardneg_offset + num_hardnegs))

            query_hardneg_indices.append((query_idx, hardneg_indices))
            all_hardnegs.extend(hardnegs)

            hardneg_offset += num_hardnegs

        query_idx += 1

    print(f"Found {len(queries)} queries")
    print(f"Found {len(all_positives)} positives")
    print(f"Found {len(query_hardneg_indices)} queries with hardnegs")
    print(f"Total hardnegs: {len(all_hardnegs)}")

    # [Step 3] Load model
    print(f"\n[Step 3] Loading model from {args.model_path}...")
    model = SentenceTransformer(args.model_path, trust_remote_code=True)
    device = [f"cuda:{i}" for i in range(args.device_count)] if args.device_count > 1 else "cuda:0"
    print(f"Model max sequence length: {model.get_max_seq_length()}")

    # [Step 4] Compute embeddings
    print("\n[Step 4] Computing embeddings...")
    cache_dir = f"../../cache/analyze_hardneg/{args.model_name}/{args.data_name}"

    # Query embeddings
    query_cache_path = os.path.join(cache_dir, "query_embeddings.npy")
    query_embeddings = load_or_compute_encodings(
        model, queries, query_cache_path,
        args.batch_size, device, args.save_encodings == 1
    )

    # Positive embeddings
    positive_cache_path = os.path.join(cache_dir, "positive_embeddings.npy")
    positive_embeddings = load_or_compute_encodings(
        model, all_positives, positive_cache_path,
        args.batch_size, device, args.save_encodings == 1
    )

    # Hardneg embeddings
    hardneg_cache_path = os.path.join(cache_dir, "hardneg_embeddings.npy")
    hardneg_embeddings = load_or_compute_encodings(
        model, all_hardnegs, hardneg_cache_path,
        args.batch_size, device, args.save_encodings == 1
    )

    # Normalize embeddings for cosine similarity
    query_embeddings = l2_normalize(query_embeddings)
    positive_embeddings = l2_normalize(positive_embeddings)
    hardneg_embeddings = l2_normalize(hardneg_embeddings)

    # [Step 5] Compute hardness metrics
    print("\n[Step 5] Computing hardness metrics...")
    metrics, similarities_qh, similarities_qp, ratios = compute_hardness_metrics(
        query_embeddings, hardneg_embeddings, positive_embeddings, query_hardneg_indices
    )

    # [Step 6] Save results
    print("\n[Step 6] Saving results...")
    output_dir = f"../../results/analyze_hardneg/{args.model_name}/{args.data_name}"
    save_results(metrics, similarities_qh, similarities_qp, ratios, output_dir)

    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()