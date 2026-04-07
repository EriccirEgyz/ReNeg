"""
Mine hard negatives from a separate dataset

This script:
1) Loads queries and positives from training data, and corpus from a separate dataset.
2) Deduplicates queries and corpus samples.
3) Encodes queries and corpus with a SentenceTransformer teacher model.
4) Loads pre-computed query-positive minimum similarities.
5) Builds a FAISS index and mines negatives per query (Naive / PercPos).
6) Writes an output jsonl with rejected_response appended.

Input format (training jsonl, one record per line):
{
  "query": "...",
  "response": "..."
}

Dataset format (jsonl, one record per line):
{
  "args.corpus_field": "..."
}

Output:
Same training records, but each record has:
  "rejected_response": ["neg1", "neg2", ...]
"""

import argparse
import os
import random
import json
from typing import Dict, List, Set, Tuple, Optional
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

    # Training data arguments
    parser.add_argument("--traindata_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--traindata_name", type=str, required=True, help="Name of the training dataset (used for caching)")
    parser.add_argument("--output_traindata_path", type=str, required=True, help="Path to save the output training data with negatives")

    # Dataset/corpus arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset to mine negatives from")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (used for caching)")
    parser.add_argument("--corpus_field", type=str, required=True, help="Field name in dataset containing corpus text")
    parser.add_argument("--special_encode_corpus_path", type=str, default="None", help="Special path for corpus embeddings (default: None)")

    # Mining arguments
    parser.add_argument("--mode", type=str, choices=["Naive", "PercPos"], required=True, help="Mining mode: 'Naive' or 'PercPos'")
    parser.add_argument("--perc_pos", type=float, required=True, help="Percentage of query-positive similarity for threshold (PercPos mode)")
    parser.add_argument("--initial_search_ratio", type=float, required=True, help="Initial search_k as a percentage of training data size, will increase if insufficient negatives found")
    parser.add_argument("--sample_upper_bound", type=int, nargs='+', required=True, help="Upper bounds of sampling ranges")
    parser.add_argument("--sample_lower_bound", type=int, nargs='+', required=True, help="Lower bounds of sampling ranges")
    parser.add_argument("--neg_number", type=int, nargs='+', required=True, help="Number of negatives to sample from each range")

    # Processing arguments
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for encoding")
    parser.add_argument("--faiss_gpu", type=int, choices=[0, 1], required=True, help="Use GPU for FAISS search (1) or CPU (0)")
    parser.add_argument("--save_encoding", type=int, choices=[0, 1], required=True, help="Save encoded embeddings (1) or not (0)")

    return parser.parse_args()

def load_training_data(
    traindata_path: str
) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    """
    Load training data and extract unique queries.

    Args:
        traindata_path: Path to the training data file

    Returns:
        Tuple of (queries, query2index, query2pos)
        - queries: List of deduplicated query strings
        - query2index: Mapping from query string to its index
        - query2pos: Mapping from query index to list of positive responses
    """
    queries = []
    queries_set = set()
    query2index = {}
    query2pos = {}

    with open(traindata_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]

            # Add new query
            if query not in queries_set:
                queries_set.add(query)
                query_idx = str(len(queries))
                query2index[query] = query_idx
                query2pos[query_idx] = []
                queries.append(query)
            else:
                query_idx = query2index[query]

            # Add positive response
            query2pos[query_idx].append(data["response"])

    print(f"Loaded {len(queries)} unique queries from training data")
    return queries, query2index, query2pos

def load_corpus_data(
    dataset_path: str,
    corpus_field: str
) -> Tuple[List[str], Dict[str, str]]:
    """
    Load corpus data from dataset.

    Args:
        dataset_path: Path to the dataset file
        corpus_field: Field name containing corpus text

    Returns:
        Tuple of (corpus, sample2index)
        - corpus: List of deduplicated corpus strings
        - sample2index: Mapping from corpus string to its index
    """
    corpus = []
    corpus_set = set()
    sample2index = {}

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sample = data[corpus_field]

            # Add new sample
            if sample not in corpus_set:
                corpus_set.add(sample)
                sample_idx = str(len(corpus))
                sample2index[sample] = sample_idx
                corpus.append(sample)

    print(f"Loaded {len(corpus)} unique corpus samples from dataset")
    return corpus, sample2index

def encode_data(
    model: SentenceTransformer,
    queries: List[str],
    corpus: List[str],
    queries_embeds_path: str,
    corpus_embeds_path: str,
    batch_size: int,
    device,
    save_encoding: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode queries and corpus using the embedding model.

    Args:
        model: SentenceTransformer model
        queries: List of query strings
        corpus: List of corpus strings
        queries_embeds_path: Path to save/load query embeddings
        corpus_embeds_path: Path to save/load corpus embeddings
        batch_size: Batch size for encoding
        device: Device to use for encoding
        save_encoding: Whether to save the encoded embeddings

    Returns:
        Tuple of (queries_embeds, corpus_embeds)
    """
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
        os.makedirs(os.path.dirname(corpus_embeds_path), exist_ok=True)
        os.makedirs(os.path.dirname(queries_embeds_path), exist_ok=True)
        np.save(corpus_embeds_path, corpus_embeds)
        np.save(queries_embeds_path, queries_embeds)
        print(f"Saved corpus embeddings to {corpus_embeds_path}")
        print(f"Saved query embeddings to {queries_embeds_path}")

    return queries_embeds, corpus_embeds

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors along axis 1."""
    norms = np.linalg.norm(vectors, ord=2, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms

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

def load_querypos_minsim(
    querypos_minsim_path: str,
    num_queries: int
) -> List[float]:
    """
    Load query-positive minimum similarities.

    Args:
        querypos_minsim_path: Path to the query-positive min sim file
        num_queries: Number of queries (for validation)

    Returns:
        List of minimum positive similarities for each query

    Raises:
        FileNotFoundError: If the querypos_minsim file does not exist
    """
    if not os.path.exists(querypos_minsim_path):
        raise FileNotFoundError(
            f"Query-positive minimum similarities file not found: {querypos_minsim_path}\n"
            "Please run other script to compute querypos_minsim first."
        )

    print(f"Loading query-positive minimum similarities from {querypos_minsim_path}")
    querypos_minsim_list = np.load(querypos_minsim_path, allow_pickle=False).tolist()

    if len(querypos_minsim_list) != num_queries:
        raise ValueError(
            f"Query-positive min sim count ({len(querypos_minsim_list)}) "
            f"does not match query count ({num_queries})"
        )

    return querypos_minsim_list

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
    search_batch_size: int = 100
) -> Dict[str, List[int]]:
    """
    Mine hard negatives for each query using FAISS search.

    Args:
        queries: List of query strings
        corpus: List of corpus strings
        queries_embeds: Query embeddings
        corpus_embeds: Corpus embeddings
        query2pos: Mapping from query index to list of positive responses
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
                pos_set = set(query2pos[query_idx])
                querypos_minsim = querypos_minsim_list[int(query_idx)]

                # Filter candidates
                filtered_cands = []
                num_acquired = 0

                if mode == "Naive":
                    for j in range(search_k):
                        if num_acquired >= k:
                            break
                        cand = int(candidates[j])

                        if corpus[cand] not in pos_set and corpus[cand] != current_query:
                            filtered_cands.append(cand)
                            num_acquired += 1

                elif mode == "PercPos":
                    negative_max_sim = querypos_minsim * perc_pos
                    for j in range(search_k):
                        if num_acquired >= k:
                            break
                        cand = int(candidates[j])

                        if (corpus[cand] not in pos_set and
                            corpus[cand] != current_query and
                            similarities[j] < negative_max_sim):
                            filtered_cands.append(cand)
                            num_acquired += 1

                # Handle insufficient candidates
                if num_acquired < k:
                    expand_search_k = True
                    if search_k == n_corpus:
                        print(
                            f"Warning: Query {query_idx} only found {num_acquired}/{k} candidates. "
                            "Consider adjusting mining settings."
                        )
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
        corpus: List of corpus strings
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

    print("=" * 70)
    print("Hard Negative Mining from Separate Dataset")
    print("=" * 70)

    # Load training data
    print("\n[Step 1] Loading training data...")
    queries, query2index, query2pos = load_training_data(args.traindata_path)

    # Load corpus data
    print("\n[Step 2] Loading corpus data...")
    corpus, sample2index = load_corpus_data(args.dataset_path, args.corpus_field)

    # Initialize model
    print("\n[Step 3] Initializing model...")
    model = SentenceTransformer(model_path, trust_remote_code=True)

    if args.device_count > 1:
        device = [f"cuda:{i}" for i in range(args.device_count)]
    else:
        device = "cuda:0"

    max_length = model.get_max_seq_length()
    print(f"Model max sequence length: {max_length}")

    # Prepare embedding paths
    encode_save_dir = f'../../cache/mine/encoding/{args.model_name}/{args.traindata_name}'
    queries_embeds_path = os.path.join(encode_save_dir, "queries_output.npy")

    if args.special_encode_corpus_path != "None":
        corpus_embeds_path = args.special_encode_corpus_path
        print(f"Using specified corpus embeddings path: {corpus_embeds_path}")
    else:
        encode_save_dir2 = f'../../results/encoding/{args.model_name}/{args.dataset_name}'
        corpus_embeds_path = os.path.join(encode_save_dir2, "corpus_output.npy")

    # Encode data
    print("\n[Step 4] Encoding queries and corpus...")
    queries_embeds, corpus_embeds = encode_data(
        model, queries, corpus, queries_embeds_path, corpus_embeds_path,
        args.batch_size, device, args.save_encoding == 1
    )

    # Normalize embeddings
    print("\n[Step 5] Normalizing embeddings...")
    queries_embeds = l2_normalize(queries_embeds)
    corpus_embeds = l2_normalize(corpus_embeds)

    # Load query-positive minimum similarities
    print("\n[Step 6] Loading query-positive minimum similarities...")
    querypos_minsim_path = f"../../cache/mine/querypos_minsim/{args.model_name}/{args.traindata_name}/querypos_minsim.npy"
    querypos_minsim_list = load_querypos_minsim(querypos_minsim_path, len(queries))

    # Build FAISS index
    print("\n[Step 7] Building FAISS index...")
    index = build_faiss_index(corpus_embeds, args.faiss_gpu == 1)

    # Mine negatives
    print("\n[Step 8] Mining hard negatives...")
    query_negs = mine_negatives(
        queries, corpus, queries_embeds, corpus_embeds,
        query2pos, querypos_minsim_list, index,
        args.mode, args.perc_pos,
        args.sample_upper_bound, args.sample_lower_bound, args.neg_number
    )

    # Write output
    print("\n[Step 9] Writing output dataset...")
    write_output_dataset(
        args.traindata_path, args.output_traindata_path,
        corpus, query2index, query_negs
    )

    print("\n" + "=" * 70)
    print("Mining completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()