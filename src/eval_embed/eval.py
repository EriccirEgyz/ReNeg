"""
Evaluate embedding models using FAISS search and TREC evaluation tools.

This script:
1) Loads a SentenceTransformer model with customizable configurations.
2) Encodes queries and corpus from a dataset in jsonl format.
3) Performs efficient similarity search using FAISS.
4) Evaluates retrieval performance using trec_eval tools.

Input format (jsonl files in dataset directory):
- queries.jsonl: {"_id": "...", "text": "..."}
- corpus.jsonl: {"_id": "...", "text": "..."}
- qrels.jsonl: {"query-id": "...", "corpus-id": "...", "score": int}

Output:
- TREC format files for evaluation
- Evaluation metrics saved to text files
"""

import argparse
import os
from typing import Dict, List, Tuple, Optional
import json
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

    # Prompt arguments
    parser.add_argument("--query_instruction", type=str, default="", help="Instruction prompt for query encoding")
    parser.add_argument("--document_instruction", type=str, default="", help="Instruction prompt for document encoding")

    # Data arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (used for caching)")
    parser.add_argument("--special_encode_corpus_path", type=str, default=None, help="Optional path to load corpus encodings for reuse across datasets")

    # Processing arguments
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for encoding")
    parser.add_argument("--stream_batch_size", type=int, default=5000000, help="Number of texts to load per batch for streaming (for memory management)")
    parser.add_argument("--faiss_gpu", type=int, choices=[0, 1], required=True, help="Use GPU for FAISS search (1) or CPU (0)")
    parser.add_argument("--search_k", type=int, required=True, help="Number of top results to retrieve per query")
    parser.add_argument("--trec_eval_m", type=int, default=-1, help="M parameter for trec_eval (-1 for no M parameter)")
    parser.add_argument("--trec_eval_path", type=str, required=True, help="Path to trec_eval executable")

    # Cache arguments
    parser.add_argument("--config_name", type=str, required=True, help="Configuration name (instruction version, truncation, etc.)")
    parser.add_argument("--save_encoding", type=int, choices=[0, 1], required=True, help="Save encoded embeddings (1) or not (0)")

    return parser.parse_args()

def load_jsonl_data(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Load data from jsonl file.

    Args:
        file_path: Path to the jsonl file

    Returns:
        Tuple of (ids list, texts list)
    """
    ids, texts = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            ids.append(str(data["_id"]))
            texts.append(data["text"])

    return ids, texts

def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    encode_save_path: Optional[str],
    encode_kwargs: Dict,
    stream_batch_size: int,
    encode_method: str,
    save_encoding: bool
) -> np.ndarray:
    """
    Encode texts using the model with caching support.

    Args:
        model: SentenceTransformer model
        texts: List of texts to encode
        encode_save_path: Path to save/load cached embeddings
        encode_kwargs: Additional arguments for encoding
        stream_batch_size: Number of texts to accumulate before encoding (for memory management)
        encode_method: Method name to call ('encode_query' or 'encode_document')
        save_encoding: Whether to save the encodings

    Returns:
        Text embeddings as numpy array
    """
    if encode_save_path and os.path.exists(encode_save_path):
        print(f"Loading cached embeddings from {encode_save_path}")
        return np.load(encode_save_path)

    print(f"Encoding {len(texts)} texts (streaming with batch size {stream_batch_size})...")
    encode_func = getattr(model, encode_method)
    embeds_list = []
    batch_texts = []

    for text in texts:
        batch_texts.append(text)
        if len(batch_texts) == stream_batch_size:
            batch_embeds = encode_func(batch_texts, **encode_kwargs).astype(np.float32)
            embeds_list.append(batch_embeds)
            batch_texts = []

    if batch_texts:
        batch_embeds = encode_func(batch_texts, **encode_kwargs).astype(np.float32)
        embeds_list.append(batch_embeds)

    embeds = np.vstack(embeds_list)

    if save_encoding and encode_save_path:
        os.makedirs(os.path.dirname(encode_save_path), exist_ok=True)
        np.save(encode_save_path, embeds)
        print(f"Saved embeddings to {encode_save_path}")

    return embeds

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

def convert_qrels(
    dataset_path: str,
    qrels_cache_path: str
) -> None:
    """
    Convert qrels.jsonl to TREC format and save to specified path.

    TREC format: query_id 0 doc_id relevance_score

    Args:
        dataset_path: Path to dataset directory containing qrels.jsonl
        qrels_cache_path: Path to save the converted qrels.trec file
    """
    print(f"Converting qrels.jsonl to TREC format...")
    os.makedirs(os.path.dirname(qrels_cache_path), exist_ok=True)

    with open(f'{dataset_path}/qrels.jsonl', 'r', encoding='utf-8') as fin, \
         open(qrels_cache_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            item = json.loads(line)
            qid = str(item["query-id"])
            docid = str(item["corpus-id"])
            score = int(item["score"])
            fout.write(f"{qid} 0 {docid} {score}\n")

    print(f"Qrels saved to {qrels_cache_path}")

def convert_run(
    run_trec_path: str,
    query_ids: List[str],
    corpus_ids: List[str],
    D: np.ndarray,
    I: np.ndarray,
    search_k: int
) -> None:
    """
    Convert FAISS search results to TREC run format and write to file.

    TREC run format: query_id Q0 doc_id rank score tag

    Args:
        run_trec_path: Path to save the run.trec file
        query_ids: List of query IDs
        corpus_ids: List of corpus/document IDs
        D: Distance/similarity scores matrix from FAISS search
        I: Index matrix from FAISS search (indices into corpus_ids)
        search_k: Number of top results per query

    Returns:
        None
    """
    print(f"Writing TREC run file to {run_trec_path}...")
    os.makedirs(os.path.dirname(run_trec_path), exist_ok=True)

    with open(run_trec_path, 'w', encoding='utf-8') as fout:
        for i, qid in enumerate(query_ids):
            for j in range(search_k):
                docno = corpus_ids[I[i, j]]
                score = float(D[i, j])
                fout.write(f"{qid} Q0 {docno} {j+1} {score} Dense\n")

    print(f"Run file saved to {run_trec_path}")

def run_trec_eval(
    qrels_trec_path: str,
    run_trec_path: str,
    result_path: str,
    trec_eval_path: str,
    trec_eval_m: int = -1
) -> None:
    """
    Run trec_eval evaluation tool.

    Args:
        qrels_trec_path: Path to qrels file in TREC format
        run_trec_path: Path to run file in TREC format
        result_path: Path to save evaluation results
        trec_eval_path: Path to trec_eval executable
        trec_eval_m: M parameter for trec_eval (-1 for no M parameter)
    """
    print(f"Running trec_eval evaluation...")

    if trec_eval_m == -1:
        cmd = f'{trec_eval_path} -m all_trec "{qrels_trec_path}" "{run_trec_path}" | tee "{result_path}"'
    else:
        print(f"Using M={trec_eval_m} for trec_eval")
        cmd = f'{trec_eval_path} -m all_trec -M {trec_eval_m} "{qrels_trec_path}" "{run_trec_path}" | tee "{result_path}"'

    exit_code = os.system(cmd)
    if exit_code != 0:
        print(f"Warning: trec_eval exited with code {exit_code}")

def main():
    """Main function."""
    args = parse_arguments()

    print("=" * 60)
    print("Embedding Model Evaluation with FAISS and TREC")
    print("=" * 60)

    # Setup cache paths
    cache_base = "../../cache/eval"
    encoding_cache_dir = os.path.join(cache_base, f"encoding/{args.model_name}/{args.dataset_name}/{args.config_name}")

    # Setup results paths
    results_dir = os.path.join("../../results/eval", f"{args.model_name}/{args.dataset_name}/{args.config_name}")

    # Encoding file paths
    if args.special_encode_corpus_path:
        print("Using special path for corpus encodings")
        corpus_embeds_path = args.special_encode_corpus_path
    else:
        corpus_embeds_path = os.path.join(encoding_cache_dir, "corpus_output.npy")

    queries_embeds_path = os.path.join(encoding_cache_dir, "queries_output.npy")

    # TREC file paths
    qrels_cache_path = os.path.join(cache_base, f"qrel_trec/{args.dataset_name}/qrels.trec")
    run_cache_path = os.path.join(cache_base, f"run_trec/{args.model_name}/{args.dataset_name}/{args.config_name}/run.trec")

    # Load model
    print("\n[Step 1] Loading model...")
    model = SentenceTransformer(args.model_path, trust_remote_code=True)
    max_length = model.get_max_seq_length()
    print(f"Model max sequence length: {max_length}")

    device = [f"cuda:{i}" for i in range(args.device_count)] if args.device_count > 1 else "cuda:0"

    # Prepare encoding kwargs
    encode_kwargs = {
        "batch_size": args.batch_size,
        "show_progress_bar": True,
        "device": device
    }

    # Load and encode corpus
    print("\n[Step 2] Loading and encoding corpus...")
    corpus_file = os.path.join(args.dataset_path, "corpus.jsonl")
    corpus_ids, corpus_texts = load_jsonl_data(corpus_file)

    encode_kwargs["prompt"] = args.document_instruction
    corpus_embeds = encode_texts(
        model, corpus_texts,
        corpus_embeds_path,
        encode_kwargs, args.stream_batch_size,
        "encode_document", args.save_encoding == 1
    )

    # Load and encode queries
    print("\n[Step 3] Loading and encoding queries...")
    queries_file = os.path.join(args.dataset_path, "queries.jsonl")
    query_ids, query_texts = load_jsonl_data(queries_file)

    encode_kwargs["prompt"] = args.query_instruction
    queries_embeds = encode_texts(
        model, query_texts,
        queries_embeds_path,
        encode_kwargs, args.stream_batch_size,
        "encode_query", args.save_encoding == 1
    )

    # Normalize embeddings
    print("\n[Step 4] Normalizing embeddings (cosine similarity)...")
    corpus_embeds = l2_normalize(corpus_embeds)
    queries_embeds = l2_normalize(queries_embeds)

    # Build FAISS index and search
    print("\n[Step 5] Building FAISS index and searching...")
    index = build_faiss_index(corpus_embeds, args.faiss_gpu == 1)

    print(f"Searching top-{args.search_k} results...")
    D, I = index.search(queries_embeds, args.search_k)

    # Convert to TREC format
    print("\n[Step 6] Converting to TREC format...")
    convert_qrels(args.dataset_path, qrels_cache_path)

    # Write run file to cache directory
    convert_run(run_cache_path, query_ids, corpus_ids, D, I, args.search_k)

    # Run trec_eval
    print("\n[Step 7] Running TREC evaluation...")
    os.makedirs(results_dir, exist_ok=True)
    if args.trec_eval_m == -1:
        result_path = os.path.join(results_dir, f"rank{args.search_k}.txt")
    else:
        result_path = os.path.join(results_dir, f"M{args.trec_eval_m}_rank{args.search_k}.txt")

    run_trec_eval(
        qrels_cache_path, run_cache_path, result_path,
        args.trec_eval_path, args.trec_eval_m
    )

    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {result_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()