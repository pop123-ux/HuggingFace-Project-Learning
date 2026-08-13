# 🤗 HuggingFace Project Learning

Hands-on work from my journey through the [Hugging Face LLM Course](https://huggingface.co/learn/nlp-course)
and [The Reasoning Course](https://huggingface.co/learn/reasoning-course), plus a
flagship end-to-end project: fine-tuning BERT for **Romanian extractive question
answering** on the XQuAD-ro dataset.

Companion to my [GitHub profile](https://github.com/pop123-ux) — the HF
completion certificates (**LLM Course Unit 1**, **Unit 3**, and **The Reasoning
Course — Fundamentals of GRPO**) shown there were earned working through the
notebooks in this repo.

---

## 🚀 Flagship Project — Romanian QA on XQuAD-ro

Two extractive QA models fine-tuned on the Romanian split of
[XQuAD](https://github.com/google-deepmind/xquad), loaded directly from the
DeepMind mirror as `xquad.ro.json` and prepared with 🤗 `datasets`.

| Notebook | Base model | Approach | Colab |
|---|---|---|---|
| [`bert-base-romanian-cased-v1.ipynb`](projects/romanian-qa-xquad-ro/bert-base-romanian-cased-v1.ipynb) | [`dumitrescustefan/bert-base-romanian-cased-v1`](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1) | Monolingual Romanian BERT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pop123-ux/HuggingFace-Project-Learning/blob/main/projects/romanian-qa-xquad-ro/bert-base-romanian-cased-v1.ipynb) |
| [`multilingual-bert.ipynb`](projects/romanian-qa-xquad-ro/multilingual-bert.ipynb) | [`bert-base-multilingual-cased`](https://huggingface.co/bert-base-multilingual-cased) | Multilingual BERT baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pop123-ux/HuggingFace-Project-Learning/blob/main/projects/romanian-qa-xquad-ro/multilingual-bert.ipynb) |

Both notebooks cover the full pipeline: dataset download → tokenization with
sliding-window context → `AutoModelForQuestionAnswering` fine-tuning → post-
processing spans → EM / F1 evaluation. See
[`projects/romanian-qa-xquad-ro/README.md`](projects/romanian-qa-xquad-ro/README.md)
for the model comparison.

### 🔭 Planned next steps

XQuAD-ro is small (~1.2k examples), and both notebooks currently fine-tune and
evaluate on that same file — enough to demonstrate the pipeline, but not a
measurement worth quoting. Two things are planned to close that gap:

- **A proper evaluation assessment.** Fine-tune on a larger Romanian QA corpus
  (or SQuAD machine-translated to Romanian) and keep XQuAD-ro fully held out, so
  the monolingual-vs-multilingual comparison rests on Exact Match / F1 scores
  measured on data neither model was trained on.
- **Publishing the models to the Hugging Face Hub.** Once the evaluation is
  trustworthy, push both fine-tuned checkpoints with model cards covering the
  training data, the held-out scores and the intended use, so they can be loaded
  straight from `from_pretrained` rather than rebuilt from the notebooks.

---

## 🧠 Reasoning & GRPO *(→ Reasoning Course certificate)*

Group Relative Policy Optimization — the RL algorithm behind reasoning models —
studied three ways: from scratch in raw PyTorch, with 🤗 `trl`, and with
`unsloth` for memory-efficient training.

| Notebook | What it covers |
|---|---|
| [`GRPO_Implementation_in_PyTorch.ipynb`](GRPO_Implementation_in_PyTorch.ipynb) | GRPO from first principles on `Qwen/Qwen2-Math-1.5B` — group sampling, reward computation, advantage normalization and the policy update, written out by hand |
| [`Fine_tuning_a_model_with_GRPO.ipynb`](Fine_tuning_a_model_with_GRPO.ipynb) | `GRPOTrainer` from `trl` + LoRA (`peft`) on `HuggingFaceTB/SmolLM-135M-Instruct` over the `mlabonne/smoltldr` summarization set, tracked in Weights & Biases |
| [`nb/Gemma3_(1B)-GRPO.ipynb`](nb/Gemma3_(1B)-GRPO.ipynb) | `google/gemma-3-1b-it` on GSM8K with `unsloth` + `vllm` — custom reward functions for answer format and correctness, then pushing the result to the Hub |

---

## 📚 HF LLM Course Notebooks

Selected chapters from the Hugging Face LLM course, reworked with my own
experiments and notes.

### Chapter 3 — Fine-tuning a pretrained model *(→ Unit 3 certificate)*
Task: GLUE / MRPC paraphrase classification with `bert-base-uncased`.

- [`01_trainer_api.ipynb`](course/chapter3-fine-tuning/01_trainer_api.ipynb) — `Trainer` API end-to-end
- [`02_full_training_loop.ipynb`](course/chapter3-fine-tuning/02_full_training_loop.ipynb) — same task with a hand-written PyTorch loop + `accelerate`
- [`03_learning_curves.ipynb`](course/chapter3-fine-tuning/03_learning_curves.ipynb) — reading and debugging training curves
- [`Evaluation.ipynb`](course/chapter3/Evaluation.ipynb) — metrics with `evaluate`

### Chapter 4 — Sharing models and tokenizers
- [`01_using_pretrained_models.ipynb`](course/chapter4-sharing-models/01_using_pretrained_models.ipynb)
- [`02_using_and_sharing.ipynb`](course/chapter4-sharing-models/02_using_and_sharing.ipynb) — pushing checkpoints to the Hub

### Chapter 5 — The 🤗 Datasets library
- [`01_importing_external_datasets.ipynb`](course/chapter5-datasets/01_importing_external_datasets.ipynb) — CSV / JSON / local files
- [`02_slicing_and_dicing.ipynb`](course/chapter5-datasets/02_slicing_and_dicing.ipynb) — `map`, `filter`, `train_test_split` on UCI drugsCom reviews
- [`03_handling_big_datasets.ipynb`](course/chapter5-datasets/03_handling_big_datasets.ipynb) — streaming + memory-mapping PubMed summarization
- [`Semantic_Search_with_FAISS.ipynb`](course/chapter5/Semantic_Search_with_FAISS.ipynb) — embedding the `lewtun/github-issues` corpus and building a FAISS index

### Chapter 6 — The 🤗 Tokenizers library
- [`Fast_Tokenizers_Special_Powers.ipynb`](course/chapter6/Fast_Tokenizers_Special_Powers.ipynb) — offset mappings, word IDs, QA/NER pipelines
- [`Training_a_New_Tokenizer_from_an_Old_One.ipynb`](course/chapter6/Training_a_New_Tokenizer_from_an_Old_One.ipynb) — retraining on `code_search_net`
- [`Normalization_and_Pre-tokenization.ipynb`](course/chapter6/Normalization_and_Pre-tokenization.ipynb) — the pieces of the tokenization pipeline
- [`Building_A_Tokenizer_from_Scratch.ipynb`](course/chapter6/Building_A_Tokenizer_from_Scratch.ipynb) — assembling one block by block
- [`WordPiece_from_Scratch.ipynb`](course/chapter6/WordPiece_from_Scratch.ipynb) — the WordPiece algorithm by hand
- [`Unigram_from_Scratch.ipynb`](course/chapter6/Unigram_from_Scratch.ipynb) — the Unigram algorithm by hand

### Chapter 7 — Classic NLP tasks
Each notebook fine-tunes and pushes a checkpoint to the Hub.

- [`TokenClassification.ipynb`](course/chapter7/TokenClassification.ipynb) — NER on CoNLL-2003 → `bert-finetuned-ner`
- [`Fine-tuning_a_Masked_Language_Model.ipynb`](course/chapter7/Fine-tuning_a_Masked_Language_Model.ipynb) — domain-adapting DistilBERT to IMDB
- [`Translation.ipynb`](course/chapter7/Translation.ipynb) — `Helsinki-NLP/opus-mt-en-fr` on KDE4 en→fr
- [`Summarization.ipynb`](course/chapter7/Summarization.ipynb) — `google/mt5-small` on multilingual Amazon reviews
- [`Question_answering.ipynb`](course/chapter7/Question_answering.ipynb) — BERT on SQuAD (the English precursor to the XQuAD-ro project above)
- [`Training_a_Data_Science_Syntax_Auto_Completer_Model(Causal Language Model)_from_Scratch.ipynb`](course/chapter7/Training_a_Data_Science_Syntax_Auto_Completer_Model%28Causal%20Language%20Model%29_from_Scratch.ipynb) — a GPT-2-style code completer trained from scratch on CodeParrot

### Chapter 8 — How to ask for help
- [`error_fix_workflow.ipynb`](course/chapter8/error_fix_workflow.ipynb) — debugging the training pipeline and reading tracebacks

### Chapter 9 — Building demos with Gradio
- [`Gradio_Interference_Class.ipynb`](course/chapter9/Gradio_Interference_Class.ipynb) — the `Interface` basics
- [`Advanced_Interface_features.ipynb`](course/chapter9/Advanced_Interface_features.ipynb) — state, interpretation, layout
- [`Playing_with_the_Gradio_API.ipynb`](course/chapter9/Playing_with_the_Gradio_API.ipynb) — driving a Space programmatically
- [`Gradio_Integration_with_HuggingFace.ipynb`](course/chapter9/Gradio_Integration_with_HuggingFace.ipynb) — wiring demos to Hub models and Spaces

### Chapter 11 — Fine-tuning LLMs
- [`Chat_Templates.ipynb`](course/chapter11/Chat_Templates.ipynb) — chat formats across SmolLM2 / Qwen / Mistral, converting `smoltalk` to model-ready text
- [`Supervised Fine-Tuning with SFTTrainer.ipynb`](course/chapter11/Supervised%20Fine-Tuning%20with%20SFTTrainer.ipynb) — `SFTTrainer` on `HuggingFaceTB/SmolLM2-135M`
- [`LoRA SFT.ipynb`](course/chapter11/LoRA_using_SFT/LoRA%20SFT.ipynb) — parameter-efficient SFT with `peft` adapters, then merging them back
- [`Evaluation.ipynb`](course/chapter11/Evaluation.ipynb) — benchmarking with `lighteval`

---

## 🛠️ Stack

**Core** `transformers` · `datasets` · `evaluate` · `accelerate` · `torch` · `tokenizers`

**Fine-tuning & RL** `trl` · `peft` · `unsloth` · `bitsandbytes` · `vllm`

**Tooling** `gradio` · `lighteval` · `faiss` · `wandb` · `scikit-learn`

## ▶️ Running locally

```bash
pip install -r requirements.txt
jupyter lab
```

`requirements.txt` covers the core course notebooks. The GRPO, SFT and
evaluation notebooks pull their heavier extras (`trl`, `peft`, `unsloth`,
`vllm`, `lighteval`, `wandb`) inline with `pip install` in the first cell, so
they run as-is in Colab.

Every notebook also carries an **Open in Colab** badge — a free GPU runtime is
enough for most course notebooks; the QA fine-tuning and GRPO notebooks want a
T4 or better.

## 🗂️ Repo layout

```
projects/   Flagship end-to-end projects (Romanian QA)
course/     HF course chapter walkthroughs (ch3–ch9, ch11)
nb/         Standalone notebooks (Gemma 3 GRPO)
*.ipynb     Reasoning course GRPO notebooks
```

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)

## 📄 License

MIT.
