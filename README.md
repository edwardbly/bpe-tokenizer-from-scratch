# BPE Tokenizer from Scratch

A byte-pair encoding (BPE) tokenizer built from first principles in Python —
no external tokenization libraries — to understand how modern language
models convert raw text into the token sequences they actually process.

## What this is

This project implements the core BPE algorithm: training a vocabulary by
iteratively merging the most frequent adjacent symbol pairs in a corpus,
then using that learned vocabulary to encode new text into tokens and
decode tokens back into text.

The notebook is organized to show the work, not just the result:

- **Sections 1–2**: the core training and encoding/decoding functions
- **Section 3**: a small, fully hand-traceable example (5 words), used to
  verify the implementation against manual predictions before trusting it
  on real data
- **Section 4**: training on a real corpus (*Alice's Adventures in
  Wonderland*, public domain via Project Gutenberg) to see what patterns
  emerge at real scale, and where the implementation's limitations show up
- **Section 5**: a comparison case — training the same tokenizer on a
  Python code sample, to test the hypothesis that code and natural
  language have different statistical structure

Each of the last three sections ends with a short "what this
revealed/confirmed" writeup — the actual findings, not just the raw output.

## Why BPE

Before a language model can process text, the text has to become a
sequence of numbers. Character-level encoding produces very long sequences
with no inherent structure; whole-word encoding requires an impossibly
large, ever-growing vocabulary. BPE sits between the two: it starts at the
character level and learns, purely from frequency in the training data,
which subword chunks are common enough to deserve their own token. Common
words often end up as a single token; rare or unfamiliar words fall back to
smaller, more granular pieces.

This project also surfaced two limitations worth being explicit about,
both documented in detail in the notebook itself:

1. **Punctuation gets glued to words.** Because word-splitting here is
   whitespace-only, `"is,"` and `"is"` are treated as unrelated tokens.
2. **Small or narrow training data causes overfitting.** Trained on a tiny,
   repetitive Python code sample, the tokenizer began memorizing entire
   function signatures character-by-character rather than learning
   generalizable patterns — a vivid demonstration of why production
   tokenizers need large, diverse training corpora.

Both are tracked as planned improvements — see
[`version2_goals.md`](version2_goals.md).

## Project status

This is a working, hand-verified v1: training, encoding, and decoding all
function correctly and have been tested against manually predicted outputs.
It is not optimized for production scale — see `version2_goals.md` for the
efficiency improvements and pre-tokenization work planned for v2.

## Running it

```bash
conda create -n tokenizer-project python=3.11
conda activate tokenizer-project
conda install jupyter
jupyter notebook
```

Open `tokenizer-project.ipynb` and run all cells. Section 4 requires
`alice.txt` (plain text of *Alice's Adventures in Wonderland*) in the same
directory — available from
[Project Gutenberg](https://www.gutenberg.org/files/11/11-0.txt).

## A note on how this was built

I used Claude (Anthropic) as a learning tool throughout this project.
Claude helped explain BPE, check my reasoning, and catch bugs. I used
Claude and Google searches to learn and verify Python syntax/behaviors.
While collaborating with Claude, I drove every design decision —
including data structures, edge case handling, and what to test and why.
Each step was carefully worked through and predicted by hand before being
confirmed in code. I can walk through and explain every function in this
notebook in detail.
