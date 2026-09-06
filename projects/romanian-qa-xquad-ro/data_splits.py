"""Leak-free train/validation/test splits for XQuAD-ro.

XQuAD ships as a single evaluation file with no training split. The original
notebooks in this project fine-tuned and evaluated on that same file, which
measures fit rather than generalization.

The fix is not simply "split it" — for extractive QA, *how* you split decides
whether the measurement means anything:

    Split by QUESTION  -> broken. XQuAD averages ~5 questions per paragraph, so
                          the model reads a context during training and is then
                          tested on a different question about that same
                          context. The passage is no longer unseen.

    Split by PARAGRAPH -> better, but paragraphs from one Wikipedia article
                          share entities, dates and phrasing, so a test
                          paragraph can still be about a topic memorised at
                          training time.

    Split by ARTICLE   -> what this module does. Every paragraph and every
                          question belonging to an article stays on one side of
                          the split, so a test article is genuinely unseen.

The split is deterministic given `seed`, and `verify_no_leakage` asserts the
property rather than assuming it.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from typing import Any

XQUAD_RO_URL = (
    "https://raw.githubusercontent.com/google-deepmind/xquad/master/xquad.ro.json"
)


def load_xquad_ro(path: str | None = None) -> dict[str, Any]:
    """Load xquad.ro.json from disk, downloading it if no path is given."""
    if path is None:
        with urllib.request.urlopen(XQUAD_RO_URL) as response:
            return json.loads(response.read().decode("utf-8"))
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_article(article: dict[str, Any], article_id: int) -> list[dict[str, Any]]:
    """Turn one nested SQuAD article into flat {question, context, answers} rows."""
    rows = []
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            if not qa.get("answers"):
                continue
            rows.append(
                {
                    "id": qa["id"],
                    "article_id": article_id,
                    "context": context,
                    "question": qa["question"],
                    "answers": {
                        "text": [a["text"] for a in qa["answers"]],
                        "answer_start": [a["answer_start"] for a in qa["answers"]],
                    },
                }
            )
    return rows


def make_splits(
    data: dict[str, Any],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split XQuAD-ro by article into train / validation / test.

    Articles — not questions — are shuffled and partitioned, so no context is
    ever shared across splits. The remainder after train_frac and val_frac
    becomes the test set.
    """
    if not 0 < train_frac + val_frac < 1:
        raise ValueError("train_frac + val_frac must leave room for a test split")

    articles = list(data["data"])
    order = list(range(len(articles)))
    random.Random(seed).shuffle(order)

    n_train = int(len(order) * train_frac)
    n_val = int(len(order) * val_frac)
    groups = {
        "train": order[:n_train],
        "validation": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }

    return {
        name: [
            row
            for article_id in ids
            for row in _flatten_article(articles[article_id], article_id)
        ]
        for name, ids in groups.items()
    }


def verify_no_leakage(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Assert that no article, context or question id crosses a split boundary.

    Raises AssertionError on any overlap. Returns a summary of the split so the
    numbers can be reported alongside results.
    """

    def digest(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    names = list(splits)
    articles = {n: {r["article_id"] for r in splits[n]} for n in names}
    contexts = {n: {digest(r["context"]) for r in splits[n]} for n in names}
    ids = {n: {r["id"] for r in splits[n]} for n in names}

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not articles[a] & articles[b], f"article overlap: {a} / {b}"
            assert not contexts[a] & contexts[b], f"context overlap: {a} / {b}"
            assert not ids[a] & ids[b], f"question id overlap: {a} / {b}"

    return {
        name: {
            "articles": len(articles[name]),
            "contexts": len(contexts[name]),
            "questions": len(splits[name]),
        }
        for name in names
    }


if __name__ == "__main__":
    splits = make_splits(load_xquad_ro())
    summary = verify_no_leakage(splits)
    print("XQuAD-ro article-level split (seed=42) — no leakage detected\n")
    print(f"{'split':<12}{'articles':>10}{'contexts':>10}{'questions':>11}")
    for name, counts in summary.items():
        print(
            f"{name:<12}{counts['articles']:>10}"
            f"{counts['contexts']:>10}{counts['questions']:>11}"
        )
    total = sum(c["questions"] for c in summary.values())
    print(f"{'total':<12}{'':>10}{'':>10}{total:>11}")
