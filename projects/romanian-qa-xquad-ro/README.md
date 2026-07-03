# Romanian QA on XQuAD-ro

Fine-tuning BERT-family models for **extractive question answering** in
Romanian, using the Romanian split of
[XQuAD](https://github.com/google-deepmind/xquad) (`xquad.ro.json`) as both
training and evaluation data.

## Notebooks

| Notebook | Base model | Notes |
|---|---|---|
| [`bert-base-romanian-cased-v1.ipynb`](bert-base-romanian-cased-v1.ipynb) | [`dumitrescustefan/bert-base-romanian-cased-v1`](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1) | Monolingual Romanian BERT — expected to benefit from a tokenizer trained on Romanian text. |
| [`multilingual-bert.ipynb`](multilingual-bert.ipynb) | [`bert-base-multilingual-cased`](https://huggingface.co/bert-base-multilingual-cased) | Multilingual baseline — same pipeline, different checkpoint. Useful as an apples-to-apples comparison. |

## Pipeline (both notebooks)

1. **Download** `xquad.ro.json` from the DeepMind XQuAD repo.
2. **Load** with `datasets.load_dataset("json", ...)` and reshape into the
   SQuAD-style `{question, context, answers: {text, answer_start}}` schema.
3. **Tokenize** with a sliding window (`stride`, `return_overflowing_tokens`,
   `return_offsets_mapping`) so long contexts survive the 384-token limit.
4. **Map** each answer span onto token positions to build `start_positions` /
   `end_positions` labels.
5. **Fine-tune** `AutoModelForQuestionAnswering` with the 🤗 `Trainer`.
6. **Post-process** logits back into text spans (n-best answers, max answer
   length).
7. **Evaluate** with `evaluate.load("squad")` → Exact Match / F1.

## Dataset caveat

XQuAD-ro is a small evaluation set (~1.2k examples). Both notebooks train and
evaluate on the same file for demonstration — a proper setup would fine-tune on
a larger Romanian QA corpus (or SQuAD translated to Romanian) and evaluate on
XQuAD-ro held out.

## Running

Open either notebook in Colab (badge inside), or locally:

```bash
pip install -r ../../requirements.txt
jupyter lab
```

A single T4 GPU is enough; each notebook trains in well under an hour.
