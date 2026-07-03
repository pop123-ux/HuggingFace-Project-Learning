# 🤗 HuggingFace Project Learning

Hands-on work from my journey through the [Hugging Face LLM Course](https://huggingface.co/learn/nlp-course)
and a flagship end-to-end project: fine-tuning BERT for **Romanian extractive
question answering** on the XQuAD-ro dataset.

Companion to my [GitHub profile](https://github.com/pop123-ux) — the HF course
completion certificates (**Unit 1**, **Unit 3**) shown there were earned working
through the notebooks in this repo.

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

---

## 📚 HF Course Notebooks

Selected chapters from the Hugging Face LLM course, reworked with my own
experiments and notes. The certificates on my profile map directly here:

> 🎓 **[Unit 1 certificate](https://github.com/pop123-ux) — Transformer basics**
> 🎓 **[Unit 3 certificate](https://github.com/pop123-ux) — Fine-tuning (Chapter 3 below)**

### Chapter 3 — Fine-tuning a pretrained model *(→ Unit 3 certificate)*
Task: GLUE / MRPC paraphrase classification with `bert-base-uncased`.

- [`01_trainer_api.ipynb`](course/chapter3-fine-tuning/01_trainer_api.ipynb) — `Trainer` API end-to-end
- [`02_full_training_loop.ipynb`](course/chapter3-fine-tuning/02_full_training_loop.ipynb) — same task with a hand-written PyTorch loop + `accelerate`
- [`03_learning_curves.ipynb`](course/chapter3-fine-tuning/03_learning_curves.ipynb) — reading and debugging training curves

### Chapter 4 — Sharing models and tokenizers
- [`01_using_pretrained_models.ipynb`](course/chapter4-sharing-models/01_using_pretrained_models.ipynb)
- [`02_using_and_sharing.ipynb`](course/chapter4-sharing-models/02_using_and_sharing.ipynb) — pushing checkpoints to the Hub

### Chapter 5 — The 🤗 Datasets library
- [`01_importing_external_datasets.ipynb`](course/chapter5-datasets/01_importing_external_datasets.ipynb) — CSV / JSON / local files
- [`02_slicing_and_dicing.ipynb`](course/chapter5-datasets/02_slicing_and_dicing.ipynb) — `map`, `filter`, `train_test_split` on UCI drugsCom reviews
- [`03_handling_big_datasets.ipynb`](course/chapter5-datasets/03_handling_big_datasets.ipynb) — streaming + memory-mapping PubMed summarization

---

## 🛠️ Stack

`transformers` · `datasets` · `evaluate` · `accelerate` · `torch` · `scikit-learn`

## ▶️ Running locally

```bash
pip install -r requirements.txt
jupyter lab
```

Every notebook also carries an **Open in Colab** badge — a free GPU runtime is
enough for the course notebooks; the QA fine-tuning notebooks want a T4 or
better.

## 🗂️ Repo layout

```
projects/   Flagship end-to-end projects (Romanian QA)
course/     HF course chapter walkthroughs (ch3, ch4, ch5)
```

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)

## 📄 License

MIT.
