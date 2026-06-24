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
  emerge at real scale — shown in before/after comparison across v1 and v2
- **Section 5**: a comparison case — training the same tokenizer on a
  Python code sample, to test the hypothesis that code and natural
  language have different statistical structure — also shown with v1/v2
  comparison
- **Section 6**: multi-source training experiment across 10 public domain
  texts spanning 1,500 years of written English, investigating new-word
  discovery rates and corpus diversity effects
- **Section 7**: observations on non-English text — training on a Japanese
  philosophical excerpt to explore how the tokenizer behaves when its
  whitespace-splitting assumption breaks down

Each section ends with a "what this revealed/confirmed" writeup — the
actual findings, not just the raw output.

## Why BPE

Before a language model can process text, the text has to become a
sequence of numbers. Character-level encoding produces very long sequences
with no inherent structure; whole-word encoding requires an impossibly
large, ever-growing vocabulary. BPE sits between the two: it starts at the
character level and learns, purely from frequency in the training data,
which subword chunks are common enough to deserve their own token. Common
words often end up as a single token; rare or unfamiliar words fall back to
smaller, more granular pieces.

## Version 2 improvements

The v1 implementation surfaced two limitations, both addressed in v2 and
documented with before/after comparisons in the notebook:

1. **Punctuation pre-tokenization.** v1's whitespace-only word-splitting
   caused punctuation to fuse with adjacent words — `"is,"` and `"is"`
   were treated as unrelated tokens, and Python code suffered runaway
   merging as entire function signatures collapsed into single tokens. v2
   adds `split_punctuation` and `split_all_punctuation`, with Unicode
   marker characters (`END_OF_WORD` / `NO_SPACE`) chosen to avoid
   collision with real text or code.
2. **Multi-source training.** v1 only accepted a single text input.
   v2 adds `merge_freq_dicts` and `train_bpe_multi`, which process one
   file at a time into a running frequency dictionary without holding all
   source text in memory simultaneously — including per-file error
   handling, optional new-word discovery tracking, and a 10-text corpus
   experiment with findings documented in Section 6.

Remaining v2 items (efficiency improvements and a real-world tokenizer
comparison) are tracked in [`version2_goals.md`](version2_goals.md).

## Project status

Version 2 is complete for its primary goals (punctuation pre-tokenization,
multi-source training). All functions are hand-verified against manually
predicted outputs. Remaining items are documented in `version2_goals.md`.

## Running it

```bash
conda create -n tokenizer-project python=3.11
conda activate tokenizer-project
conda install jupyter
jupyter notebook
```

Open `bpe_tokenizer.ipynb` and run all cells. All `.txt` corpus files are
excluded from this repository via `.gitignore` and must be downloaded
separately as described below.

### Section 4 — Alice's Adventures in Wonderland

Download the plain text file from Project Gutenberg and save it as
`alice.txt` in the project directory:

- [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11)
  (Gutenberg #11) — use the "Plain Text UTF-8" download option

### Section 6 — 10-text corpus

Download each of the following from Project Gutenberg using the "Plain Text
UTF-8" option and save with the filename shown. All Gutenberg files include
a header and footer (license text, metadata) that must be stripped before
training to reproduce the exact results documented in the notebook —
remove everything before `*** START OF THE PROJECT GUTENBERG EBOOK ***`
and after `*** END OF THE PROJECT GUTENBERG EBOOK ***`.

| Filename | Work | Gutenberg link |
|---|---|---|
| `shakespeare_clean.txt` | Complete Works of William Shakespeare | [#100](https://www.gutenberg.org/ebooks/100) |
| `middlemarch_clean.txt` | Middlemarch | [#145](https://www.gutenberg.org/ebooks/145) |
| `city_clean.txt` | The City of God, Vol. I | [#110232](https://www.gutenberg.org/ebooks/110232) |
| `moby_clean.txt` | Moby-Dick | [#2701](https://www.gutenberg.org/ebooks/2701) |
| `pride_clean.txt` | Pride and Prejudice | [#1342](https://www.gutenberg.org/ebooks/1342) |
| `frankenstein_clean.txt` | Frankenstein | [#84](https://www.gutenberg.org/ebooks/84) |
| `room_clean.txt` | A Room with a View | [#2641](https://www.gutenberg.org/ebooks/2641) |
| `alice_clean.txt` | Alice's Adventures in Wonderland | [#11](https://www.gutenberg.org/ebooks/11) |
| `romeo_clean.txt` | Romeo and Juliet (modern spelling) | [#1112](https://www.gutenberg.org/ebooks/1112) |
| `romeo2_clean.txt` | The Tragedie of Romeo and Juliet (First Folio) | [#2261](https://www.gutenberg.org/ebooks/2261) |

### Section 7 — Japanese text

The excerpt used in the notebook is the opening section of
[道徳の観念 (*Dōtoku no Kannen*, "The Concept of Morality")](https://www.aozora.gr.jp/cards/000281/files/1711.html)
by 戸坂潤 (Tosaka Jun), available on 青空文庫 (Aozora Bunko).
To experiment on the full text or other Japanese works,
[青空文庫 (Aozora Bunko)](https://www.aozora.gr.jp/) is the primary
source of public domain Japanese literature — comparable in spirit to
Project Gutenberg, with works by authors such as Natsume Sōseki,
Akutagawa Ryūnosuke, and Osamu Dazai.

**Important:** Aozora Bunko files are distributed as Shift-JIS encoded ZIP
archives by default. Before passing them to this tokenizer, extract the ZIP
and re-encode the text file to UTF-8:

```python
with open('aozora_file.txt', encoding='shift_jis') as f:
    text = f.read()
with open('aozora_file_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(text)
```

Aozora files also contain ruby annotation markup (e.g. `《》` and `｜`
characters for furigana) that you may want to strip before training,
depending on your experiment goals.

## A note on how this was built

I used Claude (Anthropic) as a learning tool throughout this project.
Claude helped explain BPE, check my reasoning, and catch bugs. I used
Claude and Google searches to learn and verify Python syntax/behaviors.
While collaborating with Claude, I drove every design decision —
including data structures, edge case handling, and what to test and why.
Each step was carefully worked through and predicted by hand before being
confirmed in code. I can walk through and explain every function in this
notebook in detail.
