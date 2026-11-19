# Legal Drafting Assistant

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/fuh2rxqjfp/badge.svg)](https://ask.thili.ai/b/fuh2rxqjfp)

This project develops a Legal Drafting Assistant designed for paralegals. It leverages an n-gram model to predict and draft simple legal sentences, which are then instantly classified into specific clause types using a fine-tuned GPT-2 model. The goal is to build an n-gram predictor from a legal corpus and integrate it with a fine-tuned GPT-2 model for clause classification, streamlining the drafting process.

## What you are building

**Goal.** You will have built a Legal Drafting Assistant that uses n-gram prediction to draft simple legal sentences and instantly classifies them into clause types with a fine-tuned GPT-2 model.

**Done when.** The project is done when you can generate legal sentences using your n-gram predictor via a CLI, and then classify those generated sentences into clause types using the integrated fine-tuned GPT-2 model, demonstrating classification accuracy on example clauses.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a small legal text corpus for n-gram training.**  
      Hint: Use text processing techniques from the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **2. Implement the n-gram next-word predictor.**  
      Hint: Apply the n-gram model building and prediction steps from the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **3. Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.**  
      Hint: Follow the steps for loading a fine-tuned GPT-2 model from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
- [ ] **4. Create a CLI interface to generate text and then classify the generated text.**  
      Hint: Integrate the n-gram predictor and the fine-tuned GPT-2 model into a command-line interface.
- [ ] **5. Evaluate classification accuracy on example clauses.**  
      Hint: Use evaluation methods for classification tasks, as demonstrated in the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Improve the n-gram predictor by implementing smoothing techniques or using higher-order n-grams.
- Enhance the CLI to allow users to specify context or starting words for the n-gram generation.
- Expand the evaluation to include more diverse legal clause examples and report precision/recall metrics.
