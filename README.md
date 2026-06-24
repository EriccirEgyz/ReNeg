# ReNeg

**Reinforcement Learning for Synthetic Hard-Negative Generation in Dense Retrieval**

ReNeg trains a small language model to generate hard negatives for dense-retriever training. It uses supervised fine-tuning (SFT) for a cold start, then applies Group Relative Policy Optimization (GRPO) with a retriever-aware reward. The reward accepts a generated passage only when it is sufficiently hard, judged irrelevant, long enough, and valid JSON.

Experiments on MS MARCO Passage and TREC-DL show that reinforcement learning improves the generator over its supervised checkpoint. In diagnostic evaluation, it reduces the false-negative rate from **20.8% to 6.9%** while retaining greater query-negative similarity than a strong teacher-mined baseline.

> This repository is a research-code snapshot accompanying the final project for *Representation and Reasoning: Foundations of Large Models*. Model and data paths are intentionally left blank in the launch scripts; configure them before running an experiment.

## Method

Given a query \(q\) and an annotated positive passage \(p^+\), the generator produces a candidate hard negative \(\tilde p^-\). ReNeg is trained in two stages:

1. **Supervised cold start.** Qwen3-8B-Base is fine-tuned on hard negatives mined by Qwen3-Embedding-4B and retained by a Qwen3-32B relevance judge.
2. **Retriever-aware RL.** The SFT checkpoint is optimized with GRPO. For the main experiment, a sample receives reward 1 only if all four conditions hold:

   \[
   r = \mathbf{1}[s_T(q,\tilde p^-) \ge 0.6\,s_T(q,p^+)]
       \mathbf{1}[J(q,p^+,\tilde p^-)=\mathrm{TN}]
       \mathbf{1}[|\tilde p^-|\ge 200]
       \mathbf{1}[\tilde p^-\in\mathcal{Y}_{\mathrm{JSON}}].
   \]

The hardness term is a threshold rather than an objective to maximize without limit. Once a passage is hard enough, optimization pressure shifts toward avoiding false negatives and malformed or degenerate outputs.

```text
MS MARCO query + positive
          |
          v
 teacher mining --> LLM-as-a-Judge filtering --> SFT generator
                                                   |
                                                   v
                        embedding reward + judge --> GRPO generator
                                                   |
                                                   v
                              generated negatives --> retriever training --> evaluation
```

### Model roles

| Role | Model used in the report |
|---|---|
| Teacher retriever / reward encoder | Qwen3-Embedding-4B |
| Hard-negative generator | Qwen3-8B-Base |
| LLM-as-a-Judge | Qwen3-32B |
| Downstream retrievers | BGE-M3-unsupervised; Qwen3-Embedding-0.6B |

## Results

All values are nDCG@10. Each query-positive pair is trained with one negative passage.

### BGE-M3-unsupervised

| Negative source | MS MARCO Dev | TREC-DL 2019 | TREC-DL 2020 |
|---|---:|---:|---:|
| Before training | 0.280 | 0.551 | 0.574 |
| Q3E4B Top 1–10 | 0.319 | 0.571 | 0.561 |
| Q3E4B Top 10–100 | 0.337 | 0.630 | **0.639** |
| Q3E4B Top 100–500 | 0.318 | 0.624 | 0.628 |
| Qwen3-8B thinking-mode generation | 0.291 | 0.578 | 0.609 |
| SFT generation | 0.330 | 0.612 | 0.561 |
| **SFT + RL generation** | **0.342** | **0.656** | 0.616 |

### Qwen3-Embedding-0.6B transfer

| Negative source | MS MARCO Dev | TREC-DL 2019 | TREC-DL 2020 |
|---|---:|---:|---:|
| Before training | 0.378 | 0.681 | 0.668 |
| SFT generation | 0.373 | 0.691 | 0.665 |
| **SFT + RL generation** | **0.387** | **0.696** | **0.679** |

### Hardness and false-negative diagnostics

| Negative source | Mean \(s(q,p^-)\) | False-negative rate |
|---|---:|---:|
| Q3E4B Top 10–100 | 0.492 | 7.1% |
| SFT generation | **0.611** | 20.8% |
| **SFT + RL generation** | 0.561 | **6.9%** |

The RL generator does not improve by merely making negatives easier: its negatives remain harder than Top 10–100 mining while becoming substantially more reliable than SFT generations.

## Repository layout

```text
ReNeg/
├── src/
│   ├── mine_neg/          # FAISS-based hard-negative mining
│   ├── llm_judge/         # SGLang judge server, filtering, and export
│   ├── sft_generator/     # SFT launch script for the generator
│   ├── rl_generator/      # GRPO launch script
│   ├── gen_embed_data/    # Batched hard-negative generation and export
│   ├── train_embed/       # BGE-M3 and Qwen3-Embedding retriever training
│   ├── eval_embed/        # FAISS retrieval and trec_eval evaluation
│   └── analyze_hardneg/   # Hardness and diversity diagnostics
└── verl/                  # Bundled veRL fork with ReNeg reward workers
```

The files `verl/verl/workers/fsdp_workers_hardneg*.py` preserve reward-function variants explored during development. The reported main reward (binary hardness threshold at 0.6, judge filtering, 200-character minimum, and JSON validation) is implemented in `fsdp_workers_hardneg9.py`.

## Setup

ReNeg is intended for Linux machines with NVIDIA GPUs. The checked-in launch configurations assume up to eight GPUs; reduce process counts, tensor parallelism, data parallelism, and batch sizes to fit your hardware.

### 1. Create an environment

Python 3.10 or newer is required by the bundled veRL package.

```bash
conda create -n reneg python=3.10 -y
conda activate reneg

# Install the bundled veRL fork and its CUDA/vLLM dependencies.
cd verl
pip install -r requirements.txt
pip install -e .
cd ..

# Components used outside veRL.
pip install ms-swift sglang sentence-transformers datasets accelerate \
  faiss-gpu scikit-learn matplotlib requests tqdm
```

Install a CUDA-compatible PyTorch build and `trec_eval` separately for your machine. For alternative veRL backends, consult the requirement files and installation notes under `verl/`.

### 2. Select the reported reward worker

The bundled veRL entry point imports `verl.workers.fsdp_workers`, while this research snapshot stores experimental workers under descriptive numbered filenames. To reproduce the main setting, create the active module from variant 9:

```bash
cp verl/verl/workers/fsdp_workers_hardneg9.py \
   verl/verl/workers/fsdp_workers.py
```

Do this before installing veRL, or reinstall the editable package afterward. Other numbered workers are ablation artifacts and do not necessarily match the reported configuration.

### 3. Configure paths

Search for empty path placeholders and replace them with local paths:

```bash
grep -RInE "=\"\"|=''|= ''|path= *$" src
```

In particular, configure:

- model paths in the mining, SFT, RL, judge-server, retriever-training, and evaluation scripts;
- input/output JSONL paths;
- the judge endpoint/model in the chosen reward worker and `src/llm_judge/**/server.sh`;
- logging directories and the optional SwanLab configuration in `src/rl_generator/rl_gen_hardneg.sh`;
- the `trec_eval` executable path in `src/eval_embed/bge-m3_eval.sh`.

Several Python programs write `../../cache` and `../../results` relative to the current directory. Run each command from its own `src/<component>/` directory, as shown below.

## Data formats

### Retriever training and mining

JSONL, one object per line:

```json
{
  "query": "Instruct: Given a web search query, retrieve relevant passages\nQuery: ...",
  "response": "the annotated positive passage",
  "rejected_response": ["hard negative 1", "hard negative 2"]
}
```

`rejected_response` is added by the miners or generation exporters. The mining input only requires `query` and `response`.

### Generation and judging

The generation utilities consume records with an ID and unprefixed query:

```json
{
  "_id": "example-id",
  "query": "web search query",
  "positive_document": "annotated positive passage"
}
```

Judge inputs additionally contain:

```json
{"negative_document": "candidate passage"}
```

The generator must return a single JSON object of the form:

```json
{"hardneg": "generated hard-negative passage"}
```

For RL, the veRL dataset must expose `query` and `positive_document` inside its `reward_model` field; see the accesses in the selected reward worker when adapting a new dataset.

## Reproducing the pipeline

The scripts are configuration templates rather than a one-command workflow. Edit the relevant paths first.

### 1. Mine teacher negatives

```bash
cd src/mine_neg
bash TrainOnlyMiner.sh
```

`TrainOnlyMiner.py` mines from positives already present in the training file. `DatasetMiner.py` provides the corresponding workflow for a separate corpus. Set `--mode PercPos --perc_pos 0.95` for positive-aware filtering, or `--mode Naive` for fixed-rank mining.

### 2. Filter SFT targets with the judge

Start the Qwen3-32B OpenAI-compatible SGLang server, then run the filtering client and exporter:

```bash
cd src/llm_judge/sft_data_filter_fn
bash server.sh                          # terminal 1
python spider_fast.py                   # terminal 2
python progress.py                      # optional progress check
python export.py
```

Configure `input_path`, `save_dir`, `ips.txt`, and `output_path` in these utilities. The exporter retains candidates judged `0` (irrelevant) and converts them into generator SFT examples.

### 3. Supervised fine-tune the generator

```bash
cd src/sft_generator
bash sft_gen_hardneg.sh
```

The reported generator starts from Qwen3-8B-Base and is trained with full-parameter SFT through `ms-swift`.

### 4. Reinforcement learning with GRPO

Start the judge server used by the reward worker, configure the SFT checkpoint, reward encoder, data, output, and logging paths, then launch:

```bash
cd src/llm_judge/rl_reward
bash server.sh                          # terminal 1

cd ../../rl_generator
bash rl_gen_hardneg.sh                  # terminal 2
```

The checked-in RL template uses 16 rollouts per prompt, a batch size of 128, one epoch, KL coefficient 0.001, and eight GPUs. The report trains on 50,000 randomly sampled MS MARCO Passage training examples.

### 5. Generate final negatives

```bash
cd src/gen_embed_data/genhardneg
bash server.sh                          # terminal 1: serve SFT or RL checkpoint
python spider_fast.py                   # terminal 2
python progress.py                      # optional
python export.py
```

### 6. Train downstream retrievers

For BGE-M3-unsupervised:

```bash
cd src/train_embed
bash st_train_bge-m3.sh
```

For Qwen3-Embedding-0.6B:

```bash
cd src/train_embed
bash swift_train_Q3E0.6B.sh
```

Both launch scripts implement contrastive training with explicit hard negatives and in-batch negatives.

### 7. Evaluate and analyze

Prepare each retrieval benchmark as a directory containing:

```text
dataset/
├── corpus.jsonl     # {"_id": ..., "text": ...}
├── queries.jsonl    # {"_id": ..., "text": ...}
└── qrels.jsonl      # {"query-id": ..., "corpus-id": ..., "score": ...}
```

Then run:

```bash
cd src/eval_embed
bash bge-m3_eval.sh

cd ../analyze_hardneg
bash analyze_hardneg_hardness.sh
bash analyze_hardneg_diversity.sh
```

Evaluation writes TREC run files and invokes `trec_eval`. The analysis scripts cache embeddings and save JSON metrics and plots under `results/analyze_hardneg/`.

## Limitations

- The experiments are in-domain: generator and retrievers are trained and evaluated in the MS MARCO passage-retrieval setting.
- Hardness and judge-estimated false-negative rate are incomplete proxies for downstream utility; the reward does not directly constrain distribution shift.
- Judge failures and incomplete MS MARCO relevance labels can introduce evaluation noise.
- The current scripts expose experiment paths and several hyperparameters through source-level placeholders, so reproducing the exact run requires careful configuration.

## Acknowledgements

ReNeg builds on [veRL](https://github.com/volcengine/verl), [ms-swift](https://github.com/modelscope/ms-swift), [SGLang](https://github.com/sgl-project/sglang), Sentence Transformers, FAISS, MS MARCO, and the Qwen3 model family.

The bundled `verl/` directory retains its upstream Apache-2.0 license and notices. No separate license is currently declared for the ReNeg-specific code; add one before redistributing it independently.