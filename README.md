# Hugging Face Project Learning
<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/2d26fdda-9823-482f-ac66-81208054e8c4" />

A worked path through the modern NLP stack — from tokenizers and `Trainer` up to LoRA, GRPO and a Romanian question-answering project — as **38 runnable notebooks, most with their outputs kept in**.

This is a learning repository, not a library. Most of it follows the [Hugging Face LLM Course](https://huggingface.co/learn/nlp-course) and [Reasoning Course](https://huggingface.co/learn/reasoning-course) with my own experiments and notes layered on; the [Romanian QA project](projects/romanian-qa-xquad-ro/) is my own end-to-end work. Every notebook opens in Colab and keeps its executed outputs, so you can read what actually happened before deciding to run anything.

**Companion Medium write-up:** [The Hugging Face API Is More Powerful Than Most Developers Realize — Here’s What You Can Actually Do With It](https://medium.com/towards-artificial-intelligence/the-hugging-face-api-is-more-powerful-than-most-developers-realize-heres-what-you-can-actually-5da841b7d44d) — the article explains the ecosystem-level reasoning behind this repository and links back here to the runnable notebooks.

---

## What you'll learn

```
foundations          model work            advanced training        projects
─────────────        ──────────────        ─────────────────        ──────────────
datasets       →     fine-tuning     →     SFT · LoRA         →     Romanian QA
tokenizers           Trainer API           GRPO / reasoning         end-to-end
evaluation           training loops        lighteval
                     Hub sharing
```

| If you want to… | Go to |
|---|---|
| Understand what a tokenizer really does | [chapter 6](course/chapter6/) — WordPiece and Unigram written from scratch |
| Fine-tune your first model | [chapter 3](course/chapter3-fine-tuning/) — `Trainer`, then the same task by hand |
| Wrangle datasets that don't fit in memory | [chapter 5](course/chapter5-datasets/) — streaming, memory-mapping, FAISS search |
| Do a real NLP task end to end | [chapter 7](course/chapter7/) — NER, translation, summarization, QA |
| Train an LLM to reason | [reasoning/](reasoning/) — GRPO from first principles, then with `trl` and `unsloth` |
| See a complete project with real results | [Romanian QA](projects/romanian-qa-xquad-ro/) |

---

## Start here

**If you're new to Hugging Face** — follow in this order. Each step assumes the one before it.

1. [`chapter3-fine-tuning/01_trainer_api.ipynb`](course/chapter3-fine-tuning/01_trainer_api.ipynb) — fine-tune BERT on paraphrase detection with `Trainer`. The shortest path to a working model.
2. [`chapter3-fine-tuning/02_full_training_loop.ipynb`](course/chapter3-fine-tuning/02_full_training_loop.ipynb) — the *same* task written as a raw PyTorch loop. This is the notebook that makes `Trainer` stop being magic.
3. [`chapter3-fine-tuning/04_evaluation_metrics.ipynb`](course/chapter3-fine-tuning/04_evaluation_metrics.ipynb) — how metrics are computed with `evaluate`.
4. [`chapter5-datasets/01_importing_external_datasets.ipynb`](course/chapter5-datasets/01_importing_external_datasets.ipynb) — getting your own data in.
5. [`chapter6/Fast_Tokenizers_Special_Powers.ipynb`](course/chapter6/Fast_Tokenizers_Special_Powers.ipynb) — offset mappings and word IDs, which you need before any QA or NER work.
6. [`chapter7/TokenClassification.ipynb`](course/chapter7/TokenClassification.ipynb) — a full task, start to finish.

**If you already know the basics** — the parts worth your time:

1. [`projects/romanian-qa-xquad-ro/`](projects/romanian-qa-xquad-ro/) — monolingual vs multilingual BERT on Romanian QA, with a result that contradicts the obvious hypothesis.
2. [`reasoning/01_grpo_from_scratch_pytorch.ipynb`](reasoning/01_grpo_from_scratch_pytorch.ipynb) — GRPO implemented by hand: group sampling, advantage normalization, policy update. No `trl`.
3. [`chapter6/WordPiece_from_Scratch.ipynb`](course/chapter6/WordPiece_from_Scratch.ipynb) and [`Unigram_from_Scratch.ipynb`](course/chapter6/Unigram_from_Scratch.ipynb) — the two algorithms implemented directly.
4. [`chapter7/causal_lm_code_completer_from_scratch.ipynb`](course/chapter7/causal_lm_code_completer_from_scratch.ipynb) — a GPT-2-style code model trained from scratch on CodeParrot.
5. [`course/chapter11/03_lora_sft.ipynb`](course/chapter11/03_lora_sft.ipynb) — LoRA adapters, then merging them back.

---

## Flagship project — Romanian QA on XQuAD-ro

**[→ Full project write-up](projects/romanian-qa-xquad-ro/)**

Two BERT encoders fine-tuned for extractive QA in Romanian under an identical recipe, to test whether a monolingual model beats a multilingual one on a lower-resource language.

| Model | Exact Match | F1 |
|---|---:|---:|
| `dumitrescustefan/bert-base-romanian-cased-v1` — monolingual | 32.34 | 45.50 |
| `bert-base-multilingual-cased` — multilingual | **36.60** | **48.46** |

Multilingual BERT won, which is not what you'd predict.

**These numbers measure fit, not generalization** — XQuAD-ro has no training split, so both notebooks fine-tune and evaluate on the same ~1.2k examples. [`03_heldout_evaluation.ipynb`](projects/romanian-qa-xquad-ro/03_heldout_evaluation.ipynb) is the corrected version: it splits **by article** so no context is shared between train and test, selects on a validation split, and scores the test split once across three seeds.

That split choice is the whole point. A naive question-level shuffle would leave **98.3% of test questions reusing a context seen during training**, because XQuAD averages ~5 questions per paragraph. The project write-up shows the measurement.

---

## Learning roadmap

The repository is ordered as a progression rather than a pile of chapters:

| Stage | What it covers | Where |
|---|---|---|
| **1. Foundations** | Fine-tuning with `Trainer`, hand-written training loops, learning curves, metrics | [`chapter3-fine-tuning/`](course/chapter3-fine-tuning/) |
| **2. Models & the Hub** | Loading pretrained checkpoints, pushing your own | [`chapter4-sharing-models/`](course/chapter4-sharing-models/) |
| **3. Data at scale** | Local files, `map`/`filter`, streaming, memory-mapping, FAISS semantic search | [`chapter5-datasets/`](course/chapter5-datasets/) |
| **4. Tokenization** | Offsets and word IDs, retraining a tokenizer, WordPiece and Unigram from scratch | [`chapter6/`](course/chapter6/) |
| **5. Classic NLP tasks** | NER, MLM domain adaptation, translation, summarization, QA, causal LM from scratch | [`chapter7/`](course/chapter7/) |
| **6. Debugging** | Reading tracebacks and fixing a broken training pipeline | [`chapter8/`](course/chapter8/) |
| **7. Demos** | Gradio interfaces, the Gradio API, wiring demos to Hub models | [`chapter9/`](course/chapter9/) |
| **8. LLM fine-tuning** | Chat templates, `SFTTrainer`, LoRA adapters, `lighteval` benchmarking | [`chapter11/`](course/chapter11/) |
| **9. Reasoning & RL** | GRPO from scratch, with `trl`, and with `unsloth` + `vllm` | [`reasoning/`](reasoning/) |
| **10. Projects** | Romanian extractive QA, monolingual vs multilingual | [`projects/`](projects/) |

---

## Reasoning & GRPO

Group Relative Policy Optimization — the RL algorithm behind reasoning models — approached three ways, deliberately in increasing order of abstraction.

| Notebook | What it covers |
|---|---|
| [`01_grpo_from_scratch_pytorch.ipynb`](reasoning/01_grpo_from_scratch_pytorch.ipynb) | GRPO from first principles on `Qwen/Qwen2-Math-1.5B` — group sampling, reward computation, advantage normalization and the policy update, written by hand |
| [`02_grpo_with_trl.ipynb`](reasoning/02_grpo_with_trl.ipynb) | The same idea through `trl`'s `GRPOTrainer` |
| [`03_grpo_finetune_smollm_lora.ipynb`](reasoning/03_grpo_finetune_smollm_lora.ipynb) | `GRPOTrainer` + LoRA on `SmolLM-135M-Instruct` over `mlabonne/smoltldr`, tracked in Weights & Biases |
| [`04_gemma3_grpo_unsloth.ipynb`](reasoning/04_gemma3_grpo_unsloth.ipynb) | `gemma-3-1b-it` on GSM8K with `unsloth` + `vllm`, custom reward functions for answer format and correctness |

Writing GRPO by hand before reaching for `GRPOTrainer` is the point of the ordering — notebook 1 is where the algorithm stops being a config object.

---

## Course notebooks

Adapted from the Hugging Face LLM Course, with my own experiments and notes. See [Attribution](#attribution) for what's mine and what isn't.

<details>
<summary><b>Chapter 3 — Fine-tuning a pretrained model</b> · GLUE/MRPC with <code>bert-base-uncased</code></summary>

- [`01_trainer_api.ipynb`](course/chapter3-fine-tuning/01_trainer_api.ipynb) — `Trainer` API end to end
- [`02_full_training_loop.ipynb`](course/chapter3-fine-tuning/02_full_training_loop.ipynb) — the same task as a hand-written PyTorch loop with `accelerate`
- [`03_learning_curves.ipynb`](course/chapter3-fine-tuning/03_learning_curves.ipynb) — reading and debugging training curves
- [`04_evaluation_metrics.ipynb`](course/chapter3-fine-tuning/04_evaluation_metrics.ipynb) — metrics with `evaluate`
</details>

<details>
<summary><b>Chapter 4 — Sharing models and tokenizers</b></summary>

- [`01_using_pretrained_models.ipynb`](course/chapter4-sharing-models/01_using_pretrained_models.ipynb)
- [`02_using_and_sharing.ipynb`](course/chapter4-sharing-models/02_using_and_sharing.ipynb) — pushing checkpoints to the Hub
</details>

<details>
<summary><b>Chapter 5 — The 🤗 Datasets library</b></summary>

- [`01_importing_external_datasets.ipynb`](course/chapter5-datasets/01_importing_external_datasets.ipynb) — CSV / JSON / local files
- [`02_slicing_and_dicing.ipynb`](course/chapter5-datasets/02_slicing_and_dicing.ipynb) — `map`, `filter`, `train_test_split` on UCI drugsCom reviews
- [`03_handling_big_datasets.ipynb`](course/chapter5-datasets/03_handling_big_datasets.ipynb) — streaming and memory-mapping PubMed
- [`04_semantic_search_faiss.ipynb`](course/chapter5-datasets/04_semantic_search_faiss.ipynb) — embedding `lewtun/github-issues` and building a FAISS index
</details>

<details>
<summary><b>Chapter 6 — The 🤗 Tokenizers library</b></summary>

- [`Fast_Tokenizers_Special_Powers.ipynb`](course/chapter6/Fast_Tokenizers_Special_Powers.ipynb) — offset mappings, word IDs, QA/NER pipelines
- [`Training_a_New_Tokenizer_from_an_Old_One.ipynb`](course/chapter6/Training_a_New_Tokenizer_from_an_Old_One.ipynb) — retraining on `code_search_net`
- [`Normalization_and_Pre-tokenization.ipynb`](course/chapter6/Normalization_and_Pre-tokenization.ipynb) — the stages of the tokenization pipeline
- [`Building_A_Tokenizer_from_Scratch.ipynb`](course/chapter6/Building_A_Tokenizer_from_Scratch.ipynb) — assembling one block by block
- [`WordPiece_from_Scratch.ipynb`](course/chapter6/WordPiece_from_Scratch.ipynb) — the WordPiece algorithm by hand
- [`Unigram_from_Scratch.ipynb`](course/chapter6/Unigram_from_Scratch.ipynb) — the Unigram algorithm by hand
</details>

<details>
<summary><b>Chapter 7 — Classic NLP tasks</b> · each fine-tunes and pushes a checkpoint</summary>

- [`TokenClassification.ipynb`](course/chapter7/TokenClassification.ipynb) — NER on CoNLL-2003
- [`Fine-tuning_a_Masked_Language_Model.ipynb`](course/chapter7/Fine-tuning_a_Masked_Language_Model.ipynb) — domain-adapting DistilBERT to IMDB
- [`Translation.ipynb`](course/chapter7/Translation.ipynb) — `Helsinki-NLP/opus-mt-en-fr` on KDE4
- [`Summarization.ipynb`](course/chapter7/Summarization.ipynb) — `google/mt5-small` on multilingual Amazon reviews
- [`Question_answering.ipynb`](course/chapter7/Question_answering.ipynb) — BERT on SQuAD; the English precursor to the Romanian project
- [`causal_lm_code_completer_from_scratch.ipynb`](course/chapter7/causal_lm_code_completer_from_scratch.ipynb) — a GPT-2-style code completer trained from scratch on CodeParrot
</details>

<details>
<summary><b>Chapters 8 & 9 — Debugging and Gradio demos</b></summary>

- [`error_fix_workflow.ipynb`](course/chapter8/error_fix_workflow.ipynb) — debugging a training pipeline
- [`Gradio_Interference_Class.ipynb`](course/chapter9/Gradio_Interference_Class.ipynb) — `Interface` basics
- [`Advanced_Interface_features.ipynb`](course/chapter9/Advanced_Interface_features.ipynb) — state, layout, interpretation
- [`Playing_with_the_Gradio_API.ipynb`](course/chapter9/Playing_with_the_Gradio_API.ipynb) — driving a Space programmatically
- [`Gradio_Integration_with_HuggingFace.ipynb`](course/chapter9/Gradio_Integration_with_HuggingFace.ipynb) — wiring demos to Hub models
</details>

<details>
<summary><b>Chapter 11 — Fine-tuning LLMs</b></summary>

- [`01_chat_templates.ipynb`](course/chapter11/01_chat_templates.ipynb) — chat formats across SmolLM2 / Qwen / Mistral
- [`02_supervised_finetuning_sfttrainer.ipynb`](course/chapter11/02_supervised_finetuning_sfttrainer.ipynb) — `SFTTrainer` on `SmolLM2-135M`
- [`03_lora_sft.ipynb`](course/chapter11/03_lora_sft.ipynb) — parameter-efficient SFT with `peft`, then merging adapters back
- [`04_evaluation_lighteval.ipynb`](course/chapter11/04_evaluation_lighteval.ipynb) — benchmarking with `lighteval`
</details>

---

## Attribution

Being clear about this matters:

| Material | Origin |
|---|---|
| `course/` | Follows the [Hugging Face LLM Course](https://huggingface.co/learn/nlp-course) curriculum. The course designs the exercises; the notebooks here are my worked versions with added experiments and notes. **Credit for the teaching material belongs to Hugging Face.** |
| `reasoning/` | Notebooks 1–3 follow the [Reasoning Course](https://huggingface.co/learn/reasoning-course); notebook 4 adapts an `unsloth` template. The from-scratch PyTorch GRPO implementation is my own working-through of the algorithm. |
| `projects/romanian-qa-xquad-ro/` | **My own project.** Task framing, dataset handling, the two-model comparison and the analysis are original work. |

The HF course certificates on [my profile](https://github.com/pop123-ux) (LLM Course Units 1 and 3, Reasoning Course — Fundamentals of GRPO) were earned working through this material.

---

## Stack

**Core** `transformers` · `datasets` · `evaluate` · `accelerate` · `torch` · `tokenizers`

**Fine-tuning & RL** `trl` · `peft` · `unsloth` · `bitsandbytes` · `vllm`

**Tooling** `gradio` · `lighteval` · `faiss` · `wandb` · `scikit-learn`

## Running

Every notebook carries an **Open in Colab** badge — the fastest way to run anything here, and a free T4 is enough for most of it. The QA fine-tuning and GRPO notebooks want a T4 or better.

Locally:

```bash
pip install -r requirements.txt
jupyter lab
```

`requirements.txt` covers the core course notebooks. The GRPO, SFT and evaluation notebooks install their heavier extras (`trl`, `peft`, `unsloth`, `vllm`, `lighteval`, `wandb`) inline in their first cell, so they run as-is in Colab.

## Repository layout

```text
projects/    End-to-end project work (Romanian QA)
reasoning/   GRPO — from scratch, with trl, with unsloth
course/      HF course chapters (3-9, 11)
```

---
## Course citation
```
@misc{huggingfacecourse,
  author = {Hugging Face},
  title = {The Hugging Face Course, 2022},
  howpublished = "\url{https://huggingface.co/course}",
  year = {2022},
  note = "[Online; accessed <today>]"
}
```
## Keep going

If this was useful, the write-ups are where the reasoning gets explained at length:

- **Companion Medium article** — [The Hugging Face API Is More Powerful Than Most Developers Realize — Here’s What You Can Actually Do With It](https://medium.com/towards-artificial-intelligence/the-hugging-face-api-is-more-powerful-than-most-developers-realize-heres-what-you-can-actually-5da841b7d44d)
- **Medium profile** — [medium.com/@Pop123](https://medium.com/@Pop123)
- **Hugging Face** — [pop123ux](https://huggingface.co/pop123ux)
- **GitHub** — [@pop123-ux](https://github.com/pop123-ux), where the [LeNet-5 from scratch](https://github.com/pop123-ux/LeNet_5-from-scratch) project takes the same approach to a 1998 CNN paper

Corrections and questions are welcome via issues — particularly on the Romanian QA evaluation, which is the part most in need of a second pair of eyes.

## License

MIT — see [LICENSE](LICENSE). Course-derived material remains subject to the [Hugging Face course's own licensing](https://github.com/huggingface/course).
