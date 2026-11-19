"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import math
import pytest

# Milestone 1: Prepare a small legal text corpus for n-gram training.
def test_milestone_1():
    from app.core import milestone_1
    from app.corpus import clean_text, tokenize, split_sentences

    # 1. Verify text cleaning
    raw = "  “Confidential   Information” — shall remain strictly private.  "
    cleaned = clean_text(raw)
    assert '"Confidential Information" - shall remain strictly private.' == cleaned

    # 2. Verify tokenization
    tokens = tokenize("This Agreement shall be governed by law.", lowercase=True)
    assert tokens == ["this", "agreement", "shall", "be", "governed", "by", "law"]

    # 3. Verify sentence splitting
    sentences = split_sentences("First sentence. Second sentence! Third sentence?")
    assert len(sentences) == 3

    # 4. Verify corpus preparation via milestone_1
    result = milestone_1()
    assert result["sentence_count"] >= 30
    assert result["total_tokens"] > 300
    assert result["vocab_size"] > 100
    assert len(result["tokenized_sentences"]) == result["sentence_count"]
    assert "agreement" in result["vocabulary"]
    assert "confidential" in result["vocabulary"]
    assert "shall" in result["vocabulary"]



# Milestone 2: Implement the n-gram next-word predictor.
def test_milestone_2():
    from app.core import milestone_2
    from app.ngram import NGramPredictor

    # 1. Direct NGramPredictor unit tests (bigram and trigram)
    sentences = [
        ["this", "agreement", "shall", "be", "governed", "by", "law"],
        ["this", "agreement", "may", "be", "terminated", "by", "notice"],
        ["either", "party", "may", "terminate", "this", "agreement"],
    ]

    bigram = NGramPredictor(n=2, smoothing="laplace", k=1.0)
    bigram.fit(sentences)

    # Context "this" -> "agreement" should have high probability
    preds = bigram.predict_next_word("this", top_k=3)
    words = [w for w, _ in preds]
    assert "agreement" in words

    # Test generation from seed
    draft = bigram.generate(seed="this agreement", max_tokens=8, seed_random=42)
    assert draft.startswith("This agreement")
    assert draft.endswith(".")

    # Test perplexity calculation
    ppl = bigram.perplexity(sentences)
    assert ppl > 0 and not math.isinf(ppl)

    # 2. Test milestone_2 end-to-end function
    res = milestone_2(n=3)
    assert "predictor" in res
    assert res["n"] == 3
    assert len(res["prompts"]) > 0
    assert "This Agreement shall" in res["generations"]
    assert len(res["predictions"]["This Agreement shall"]) > 0



# Milestone 3: Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Create a CLI interface to generate text and then classify the generated text.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Evaluate classification accuracy on example clauses.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

