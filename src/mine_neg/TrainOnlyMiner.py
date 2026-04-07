"""
Mine hard negatives from training corpus only

This script:
1) Deduplicates queries and responses from a jsonl training set.
2) Encodes queries and corpus with a SentenceTransformer teacher model.
3) Builds a FAISS index and mines negatives per query (Naive / PercPos).
4) Writes an output jsonl with rejected_response appended.

Input format (jsonl, one record per line):
{
  "query": "...",
  "response": "..."
}

Output:
Same records, but each record has:
  "rejected_response": ["neg1", "neg2", ...]
"""

import argparse
import os
import random
import json
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--device_count", type=int, required=True, help="Number of GPU devices to use")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the embedding model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (used for caching)")

    # Data arguments
    parser.add_argument("--traindata_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--traindata_name", type=str, required=True, help="Name of the training dataset (used for caching)")
    parser.add_argument("--output_traindata_path", type=str, required=True, help="Path to save the output training data with negatives")

    # Mining arguments
    parser.add_argument("--mode", type=str, choices=["Naive", "PercPos"], required=True, help="Mining mode: 'Naive' or 'PercPos'")
    parser.add_argument("--perc_pos", type=float, default=0.95, help="Percentage of query-positive similarity for threshold (PercPos mode)")
    parser.add_argument("--initial_search_ratio", type=float, required=True, help="Initial search_k as a percentage of training data size, will increase if insufficient negatives found")
    parser.add_argument("--sample_upper_bound", type=int, nargs='+', required=True, help="Upper bounds of sampling ranges")
    parser.add_argument("--sample_lower_bound", type=int, nargs='+', required=True, help="Lower bounds of sampling ranges")
    parser.add_argument("--neg_number", type=int, nargs='+', required=True, help="Number of negatives to sample from each range")

    # Processing arguments
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for encoding")
    parser.add_argument("--faiss_gpu", type=int, choices=[0, 1], required=True, help="Use GPU for FAISS search (1) or CPU (0)")
    parser.add_argument("--save_encoding", type=int, choices=[0, 1], required=True, help="Save encoded embeddings (1) or not (0)")
    parser.add_argument("--save_querypos_minsim", type=int, choices=[0, 1], required=True, help="Save query-positive minimum similarities (1) or not (0)")

    return parser.parse_args()

def load_and_deduplicate_data(
    traindata_path: str
) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, List[str]]]:
    """
    Load training data and deduplicate queries and responses.

    Args:
        traindata_path: Path to the training data file

    Returns:
        Tuple of (queries, corpus, query2index, query2pos)
        - queries: List of deduplicated query strings
        - corpus: List of deduplicated response strings
        - query2index: Mapping from query string to its index
        - query2pos: Mapping from query index to list of positive response indices
    """
    queries = []
    corpus = []
    queries_set = set()
    corpus_set = set()
    query2index = {}
    sample2index = {}
    query2pos = {}

    with open(traindata_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]
            response = data["response"]

            # Process query
            if query not in queries_set:
                queries_set.add(query)
                query_idx = str(len(queries))
                query2index[query] = query_idx
                query2pos[query_idx] = []
                queries.append(query)
            else:
                query_idx = query2index[query]

            # Process response
            if response not in corpus_set:
                corpus_set.add(response)
                response_idx = str(len(corpus))
                sample2index[response] = response_idx
                corpus.append(response)
            else:
                response_idx = sample2index[response]

            query2pos[query_idx].append(response_idx)

    print(f"Loaded {len(queries)} unique queries and {len(corpus)} unique responses")
    return queries, corpus, query2index, query2pos

def encode_data(
    model: SentenceTransformer,
    queries: List[str],
    corpus: List[str],
    encode_save_dir: str,
    batch_size: int,
    device: str,
    save_encoding: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode queries and corpus using the embedding model.

    Args:
        model: SentenceTransformer model
        queries: List of query strings
        corpus: List of response strings
        encode_save_dir: Directory to save/load cached embeddings
        batch_size: Batch size for encoding
        device: Device to use for encoding
        save_encoding: Whether to save the encoded embeddings

    Returns:
        Tuple of (queries_embeds, corpus_embeds)
    """
    corpus_embeds_path = os.path.join(encode_save_dir, "corpus_output.npy")
    queries_embeds_path = os.path.join(encode_save_dir, "queries_output.npy")

    # Encode corpus
    if os.path.exists(corpus_embeds_path):
        print(f"Loading cached corpus embeddings from {corpus_embeds_path}")
        corpus_embeds = np.load(corpus_embeds_path)
    else:
        print(f"Encoding {len(corpus)} corpus samples...")
        corpus_embeds = model.encode_document(
            corpus,
            prompt="",
            batch_size=batch_size,
            show_progress_bar=True,
            device=device
        ).astype(np.float32)

    # Encode queries
    if os.path.exists(queries_embeds_path):
        print(f"Loading cached query embeddings from {queries_embeds_path}")
        queries_embeds = np.load(queries_embeds_path)
    else:
        print(f"Encoding {len(queries)} queries...")
        queries_embeds = model.encode_query(
            queries,
            prompt="",
            batch_size=batch_size,
            show_progress_bar=True,
            device=device
        ).astype(np.float32)

    # Save embeddings if requested
    if save_encoding:
        os.makedirs(encode_save_dir, exist_ok=True)
        np.save(corpus_embeds_path, corpus_embeds)
        np.save(queries_embeds_path, queries_embeds)
        print(f"Saved embeddings to {encode_save_dir}")

    return queries_embeds, corpus_embeds

def compute_querypos_min_sim(
    queries_embeds: np.ndarray,
    corpus_embeds: np.ndarray,
    query2pos: Dict[str, List[str]],
    save_dir: str,
    save_querypos_minsim: bool
) -> List[float]:
    """
    Compute minimum similarity between each query and its positive responses.

    Args:
        queries_embeds: Query embeddings
        corpus_embeds: Response embeddings
        query2pos: Mapping from query index to positive response indices
        save_dir: Directory to save the computed similarities
        save_querypos_minsim: Whether to save the computed similarities

    Returns:
        List of minimum positive similarities for each query
    """
    querypos_minsim_list = []

    for i, query_embed in enumerate(queries_embeds):
        query_idx = str(i)
        positives = query2pos[query_idx]

        # Find minimum similarity with positive responses
        querypos_minsim = 1.0
        for positive_idx in positives:
            positive_embed = corpus_embeds[int(positive_idx)]
            sim = np.dot(positive_embed, query_embed)
            querypos_minsim = min(querypos_minsim, sim)

        querypos_minsim_list.append(querypos_minsim)

    # Save results if requested
    if save_querypos_minsim:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "querypos_minsim.npy")
        np.save(save_path, querypos_minsim_list)
        print(f"Saved query-positive minimum similarities to {save_path}")
    else:
        print("Skipping save of query-positive minimum similarities")

    return querypos_minsim_list

def build_faiss_index(
    corpus_embeds: np.ndarray,
    use_gpu: bool
) -> faiss.Index:
    """
    Build FAISS index for efficient similarity search.

    Args:
        corpus_embeds: Normalized corpus embeddings
        use_gpu: Whether to use GPU for FAISS

    Returns:
        FAISS index
    """
    dim = corpus_embeds.shape[-1]
    cpu_index = faiss.IndexFlatIP(dim)

    if use_gpu:
        ngpus = faiss.get_num_gpus()
        print(f"Building FAISS index on {ngpus} GPU(s)...")
        index = faiss.index_cpu_to_all_gpus(cpu_index)
    else:
        print("Building FAISS index on CPU...")
        index = cpu_index

    index.add(corpus_embeds)
    return index

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors along axis 1."""
    norms = np.linalg.norm(vectors, ord=2, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms

def sample_negatives_from_ranges(
    filtered_cands: List[int],
    num_acquired: int,
    sample_upper_bound: List[int],
    sample_lower_bound: List[int],
    neg_number: List[int]
) -> List[int]:
    """
    Sample negatives from predefined ranges.

    Args:
        filtered_cands: List of candidate indices
        num_acquired: Number of candidates acquired
        sample_upper_bound: Upper bounds for each range
        sample_lower_bound: Lower bounds for each range
        neg_number: Number of negatives to sample from each range

    Returns:
        List of selected negative indices
    """
    final_cands = []

    for j in range(len(sample_upper_bound)):
        # Get candidates from the specified range
        if num_acquired >= sample_upper_bound[j]:
            current_cands = filtered_cands[(sample_lower_bound[j] - 1):sample_upper_bound[j]]
        elif num_acquired >= sample_lower_bound[j]:
            current_cands = filtered_cands[(sample_lower_bound[j] - 1):num_acquired]
        else:
            current_cands = []

        # Sample from candidates
        if len(current_cands) >= neg_number[j]:
            selected = random.sample(current_cands, neg_number[j])
        else:
            selected = current_cands

        final_cands.extend(selected)

    return final_cands

def mine_negatives(
    queries: List[str],
    corpus: List[str],
    queries_embeds: np.ndarray,
    corpus_embeds: np.ndarray,
    query2pos: Dict[str, List[str]],
    querypos_minsim_list: List[float],
    index: faiss.Index,
    mode: str,
    perc_pos: float,
    initial_search_ratio: float,
    sample_upper_bound: List[int],
    sample_lower_bound: List[int],
    neg_number: List[int],
    search_batch_size: int = 1000
) -> Dict[str, List[int]]:
    """
    Mine hard negatives for each query using FAISS search.

    Args:
        queries: List of query strings
        corpus: List of response strings
        queries_embeds: Query embeddings
        corpus_embeds: Response embeddings
        query2pos: Mapping from query index to positive response indices
        querypos_minsim_list: Minimum positive similarity for each query
        index: FAISS index for search
        mode: Mining mode ('Naive' or 'PercPos')
        perc_pos: Percentage for similarity threshold (PercPos mode)
        sample_upper_bound: Upper bounds for sampling ranges
        sample_lower_bound: Lower bounds for sampling ranges
        neg_number: Number of negatives to sample from each range
        search_batch_size: Batch size for FAISS search

    Returns:
        Dictionary mapping query indices to lists of negative indices
    """
    n_corpus = len(corpus)
    k = max(sample_upper_bound)
    n_queries = len(queries)

    print(f"Mining negatives with k={k}, mode={mode}")
    query_negs = {}

    for start in tqdm(range(0, n_queries, search_batch_size), desc="Mining negatives"):
        end = min(start + search_batch_size, n_queries)
        batch_q_emb = queries_embeds[start:end]

        search_k = max(1, int(n_corpus * initial_search_ratio))

        while search_k <= n_corpus:
            expand_search_k = False

            # Search for similar candidates
            D, I = index.search(batch_q_emb, search_k)

            for i in range(end - start):
                query_idx = str(start + i)

                if query_idx in query_negs:
                    continue

                current_query = queries[int(query_idx)]
                candidates = I[i]
                similarities = D[i]

                # Get positive samples to exclude
                pos_set = set(int(idx) for idx in query2pos[query_idx])
                querypos_minsim = querypos_minsim_list[int(query_idx)]

                # Filter candidates
                filtered_cands = []
                num_acquired = 0

                if mode == "Naive":
                    for j in range(search_k):
                        if num_acquired >= k:
                            break
                        cand = int(candidates[j])

                        if cand not in pos_set and corpus[cand] != current_query:
                            filtered_cands.append(cand)
                            num_acquired += 1

                elif mode == "PercPos":
                    negative_max_sim = querypos_minsim * perc_pos
                    for j in range(search_k):
                        if num_acquired >= k:
                            break
                        cand = int(candidates[j])

                        if (cand not in pos_set and
                            corpus[cand] != current_query and
                            similarities[j] < negative_max_sim):
                            filtered_cands.append(cand)
                            num_acquired += 1

                # Handle insufficient candidates
                if num_acquired < k:
                    expand_search_k = True
                    if search_k == n_corpus:
                        print(f"Warning: Query {query_idx} only found {num_acquired}/{k} candidates")
                        final_cands = sample_negatives_from_ranges(
                            filtered_cands, num_acquired,
                            sample_upper_bound, sample_lower_bound, neg_number
                        )
                        query_negs[query_idx] = final_cands
                        expand_search_k = False
                else:
                    final_cands = sample_negatives_from_ranges(
                        filtered_cands, num_acquired,
                        sample_upper_bound, sample_lower_bound, neg_number
                    )
                    query_negs[query_idx] = final_cands

            # Expand search if needed
            if expand_search_k and search_k < n_corpus:
                search_k = min(search_k * 4, n_corpus)
            else:
                break

    print(f"Mined negatives for {len(query_negs)} queries")
    return query_negs

def write_output_dataset(
    traindata_path: str,
    output_path: str,
    corpus: List[str],
    query2index: Dict[str, str],
    query_negs: Dict[str, List[int]]
) -> None:
    """
    Write output dataset with rejected responses.

    Args:
        traindata_path: Path to input training data
        output_path: Path to save output data
        corpus: List of response strings
        query2index: Mapping from query string to index
        query_negs: Mapping from query index to negative indices
    """
    with open(traindata_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        current_query_idx = ""
        current_negatives = []

        for line in fin:
            data = json.loads(line)
            query = data["query"]

            query_idx = query2index[query]

            # Update negatives when query changes
            if query_idx != current_query_idx:
                current_query_idx = query_idx
                neg_ids = query_negs.get(query_idx, [])
                current_negatives = [corpus[int(neg_id)] for neg_id in neg_ids]

            # Add rejected responses
            if "rejected_response" in data:
                data["rejected_response"].extend(current_negatives)
            else:
                data["rejected_response"] = current_negatives

            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Saved output dataset to {output_path}")

def main():
    """Main function."""
    args = parse_arguments()

    print("=" * 60)
    print("Hard Negative Mining for Query-Response Training Data")
    print("=" * 60)

    # Load and deduplicate data
    print("\n[Step 1] Loading and deduplicating training data...")
    queries, corpus, query2index, query2pos = load_and_deduplicate_data(args.traindata_path)

    # Initialize model
    print(f"\n[Step 2] Loading model from {args.model_path}...")
    model = SentenceTransformer(args.model_path, trust_remote_code=True)
    device = [f"cuda:{i}" for i in range(args.device_count)] if args.device_count > 1 else "cuda:0"

    max_length = model.get_max_seq_length()
    print(f"Model max sequence length: {max_length}")

    # Encode data
    print("\n[Step 3] Encoding queries and responses...")
    encode_save_dir = f'../../cache/mine/encoding/{args.model_name}/{args.traindata_name}'
    queries_embeds, corpus_embeds = encode_data(
        model, queries, corpus, encode_save_dir,
        args.batch_size, device, args.save_encoding == 1
    )

    # Normalize embeddings
    queries_embeds = l2_normalize(queries_embeds)
    corpus_embeds = l2_normalize(corpus_embeds)

    # Compute query-positive minimum similarities
    print("\n[Step 4] Computing query-positive minimum similarities...")
    querypos_minsim_dir = f"../../cache/mine/querypos_minsim/{args.model_name}/{args.traindata_name}"
    querypos_minsim_list = compute_querypos_min_sim(
        queries_embeds, corpus_embeds, query2pos, querypos_minsim_dir, args.save_querypos_minsim == 1
    )

    # Build FAISS index
    print("\n[Step 5] Building FAISS index...")
    index = build_faiss_index(corpus_embeds, args.faiss_gpu == 1)

    # Mine negatives
    print("\n[Step 6] Mining hard negatives...")
    query_negs = mine_negatives(
        queries, corpus, queries_embeds, corpus_embeds,
        query2pos, querypos_minsim_list, index,
        args.mode, args.perc_pos, args.initial_search_ratio,
        args.sample_upper_bound, args.sample_lower_bound, args.neg_number
    )

    # Write output
    print("\n[Step 7] Writing output dataset...")
    write_output_dataset(
        args.traindata_path, args.output_traindata_path,
        corpus, query2index, query_negs
    )

    print("\n" + "=" * 60)
    print("Mining completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()