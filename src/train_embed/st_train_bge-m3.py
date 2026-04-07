import argparse, os, sys, json, random
from datasets import load_dataset, Dataset, DatasetDict
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from transformers.integrations import WandbCallback

def load(jsonl_path, hardneg_number, q_instruct, p_instruct) -> Dataset:
    anchors, positives = [], []
    neg_cols = {f"negative_{i+1}": [] for i in range(hardneg_number)}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            q_with_instruction = o.get("query")
            q = q_with_instruction.split("Query:")[-1].strip()
            q = q_instruct + q
            p = o.get("response")
            p = p_instruct + p
            anchors.append(q)
            positives.append(p)

            if hardneg_number > 0:
                negs = o.get("rejected_response")
                negs = negs[:hardneg_number]
                for i in range(hardneg_number):
                    neg_cols[f"negative_{i+1}"].append(p_instruct + negs[i])

    data = {"anchor": anchors, "positive": positives, **neg_cols}
    return Dataset.from_dict(data)

def split(ds: Dataset, seed: int = 42, eval_size=0.02) -> DatasetDict:
    n = len(ds); idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_eval = int(round(n * eval_size)) if isinstance(eval_size, float) else int(eval_size)
    n_eval = max(1, min(n_eval, n - 1))
    return ds.select(idx[n_eval:]), ds.select(idx[:n_eval])  # train, eval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--dataset", type=str, nargs='+', help="Training dataset path(s), supports multiple datasets (space-separated)")
    parser.add_argument("--hardneg_number", type=int)
    parser.add_argument("--query_instruction", type=str)
    parser.add_argument("--passage_instruction", type=str)
    parser.add_argument("--split_dataset_ratio", type=float)
    parser.add_argument("--gather_across_devices", type=bool)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--per_device_eval_batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--run_name", type=str)

    args = parser.parse_args()

    # Load a model to finetune
    model = SentenceTransformer(
        args.model,
    )
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    # Load training datasets
    print("[WARN] Currently only supports datasets with the same number of rejected responses")
    print("[WARN] Query instruction is expected to end with 'Query:'")
    print(f"[INFO] Loading {len(args.dataset)} dataset(s)...")

    # Load all datasets
    datasets_list = []
    for i, dataset_path in enumerate(args.dataset):
        print(f"[INFO] Loading dataset {i+1}/{len(args.dataset)}: {dataset_path}")
        ds = load(dataset_path,
            hardneg_number=args.hardneg_number,
            q_instruct=args.query_instruction,
            p_instruct=args.passage_instruction)
        datasets_list.append(ds)
        print(f"[INFO] Dataset {i+1} loaded, containing {len(ds)} samples")

    # Merge all datasets
    if len(datasets_list) == 1:
        ds = datasets_list[0]
    else:
        print(f"[INFO] Merging {len(datasets_list)} datasets...")
        from datasets import concatenate_datasets
        ds = concatenate_datasets(datasets_list)
        print(f"[INFO] Merge complete, total {len(ds)} samples")

    train_ds, eval_ds = split(ds, seed=42, eval_size=args.split_dataset_ratio)

    # Define loss function
    loss = MultipleNegativesRankingLoss(model=model,
        scale=20.0,
        gather_across_devices=args.gather_across_devices)

    # Specify training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=5,
        dataloader_drop_last=True,
        run_name=args.run_name
    )

    # Create trainer and start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
        callbacks=[WandbCallback()]
    )
    trainer.train()

    # Save the trained model
    save_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(save_path)

if __name__ == "__main__":
    main()