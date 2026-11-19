"""Your code. One stub per milestone; build them in order and keep each one small."""

from __future__ import annotations

from typing import Any
from app.corpus import clean_text, tokenize, split_sentences, prepare_corpus, DEFAULT_LEGAL_CORPUS

# --- Milestone 1: Prepare a small legal text corpus for n-gram training.
# Hint: Use text processing techniques from the 'Build a Next-Word Predictor from Scratch' drill.
def milestone_1(corpus_path: str | None = None) -> dict[str, Any]:
    """Prepare and tokenize the legal corpus for n-gram training.
    
    Processes the legal text corpus with normalization, tokenization,
    vocabulary extraction, and frequency counting.
    """
    processed = prepare_corpus(corpus_path or "data/legal_corpus.txt", lowercase=True)
    return {
        "sentence_count": len(processed["raw_sentences"]),
        "total_tokens": processed["total_tokens"],
        "vocab_size": processed["vocab_size"],
        "vocabulary": processed["vocabulary"],
        "tokenized_sentences": processed["tokenized_sentences"],
        "token_frequencies": processed["token_frequencies"],
    }



from app.ngram import NGramPredictor

# --- Milestone 2: Implement the n-gram next-word predictor.
# Hint: Apply the n-gram model building and prediction steps from the 'Build a Next-Word Predictor from Scratch' drill.
def milestone_2(n: int = 3, smoothing: str = "laplace") -> dict[str, Any]:
    """Train an n-gram next-word predictor on the legal corpus and generate sample drafts.
    
    Returns the trained predictor model, next-word predictions for legal prompts,
    and generated legal sentences.
    """
    corpus_info = milestone_1()
    predictor = NGramPredictor(n=n, smoothing=smoothing, k=1.0)
    predictor.fit(corpus_info["tokenized_sentences"])

    prompts = [
        "This Agreement shall",
        "Either party may terminate",
        "The Receiving Party agrees to",
        "The Service Provider agrees to defend",
    ]

    predictions: dict[str, list[tuple[str, float]]] = {}
    generations: dict[str, str] = {}

    for prompt in prompts:
        predictions[prompt] = predictor.predict_next_word(prompt, top_k=5)
        generations[prompt] = predictor.generate(seed=prompt, max_tokens=15, temperature=0.7, seed_random=42)

    return {
        "predictor": predictor,
        "n": n,
        "smoothing": smoothing,
        "vocab_size": len(predictor.vocab),
        "prompts": prompts,
        "predictions": predictions,
        "generations": generations,
    }



# --- Milestone 3: Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.
# Hint: Follow the steps for loading a fine-tuned GPT-2 model from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Create a CLI interface to generate text and then classify the generated text.
# Hint: Integrate the n-gram predictor and the fine-tuned GPT-2 model into a command-line interface.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Evaluate classification accuracy on example clauses.
# Hint: Use evaluation methods for classification tasks, as demonstrated in the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
