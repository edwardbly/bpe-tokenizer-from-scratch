# BPE Tokenizer — Version 2 Goals

Captured after building and testing the v1 tokenizer on:
- A 5-word toy example (hand-traced)
- Alice's Adventures in Wonderland (full book, target_vocab_size=500)
- A short Python code sample (3 functions, target_vocab_size=150)

## 1. Pre-tokenization: split punctuation from words before BPE runs — ✅ DONE

**Problem observed:** our current `get_word_freqs` only splits on whitespace
(`text.split()`), so punctuation stays glued to words.

Evidence from Alice in Wonderland training:
- `('i', 's,_')` — "is," and "is" are treated as unrelated tokens
- `('k', ',_')`, `('h', ',_')` — commas fused onto word-ending letters

Evidence from Python code training (more severe):
- `('get_pair_freqs(symbol_freqs', '):_')` — an entire function signature
  collapsed into one token
- `('tuple(list(word', ')_')` — parens and method calls fused into identifiers

**What was built:** `split_punctuation` and `split_all_punctuation`, wired
into `get_word_freqs`, repeatedly split any word on the first punctuation
character found (using `string.punctuation`) until no punctuation remains.
A `NO_SPACE` marker (Unicode WORD JOINER, `\u2060`) is attached to split-off
pieces so the original spacing is exactly preserved on decode — e.g. "is,"
correctly reconstructs from two independent tokens, "is" and ",", with no
stray space inserted between them.

**Resolved open question (apostrophes):** tested treating the apostrophe as
ordinary punctuation, split the same as everything else, rather than a
special case. Decided against special-casing it after recognizing the same
character (`'`) is used both for English possessives/contractions *and* as
a Python string delimiter — a context-blind special case would help one
domain (prose) while actively breaking the other (code). Confirmed via
testing that common words still merge correctly regardless of how an
individual instance was originally split (e.g. "Alice" reliably forms the
same way whether standalone or extracted from "Alice's"), since BPE's
frequency-driven merging makes this self-correcting — no special-casing
needed.

**Verified outcome:** re-ran training on both Alice in Wonderland and the
Python code sample with the fix in place. Punctuation now merges as its own
free-standing, reusable token (e.g. `('¤,', '▁')` for a standalone comma)
rather than fusing to adjacent words. The Python code runaway-merging
failure mode is gone — training now stops early ("no more pairs to merge")
on clean, recognizable identifiers (`tuple`, `list`, `range`, `len`,
`symbol`) instead of continuing to manufacture larger tokens by absorbing
syntax characters. Full before/after comparison documented in the notebook,
Sections 4 and 5.

**Marker character design note:** also discovered during this work that the
original `'_'` end-of-word marker collided with real underscores in Python
variable names (e.g. `word_freqs`). Replaced with `END_OF_WORD` (`\u2581`,
matching SentencePiece's convention) and `NO_SPACE` (`\u2060`, Unicode WORD
JOINER) — both chosen specifically to avoid colliding with characters that
could legitimately appear in real text or code. Configurable debug-mode
substitutes (`§`, `¤`) are available for human-readable output during
inspection, independently toggleable per marker since `END_OF_WORD` already
renders clearly while the real `NO_SPACE` character prints as an unreadable
escape code.

## 2. Efficiency: avoid rebuilding the entire dataset on every merge

**Problem observed:** `merge_pair` currently iterates through every word in
`symbol_freqs` on every single merge iteration, even when the vast majority
of words don't contain the pair being merged. Confirmed as a real (if small)
slowdown once we moved from toy data to a full book.

**Goal:** only touch/rebuild words that actually contain the pair being
merged on a given iteration, rather than rebuilding the whole dictionary
every time. Likely approach: maintain an index of "which words contain which
pairs" so we can look up affected words directly instead of scanning
everything.

## 3. Training data scale vs. target_vocab_size

**Problem observed:** training on a tiny, repetitive code sample
(3 short functions, ~150 target vocab) caused runaway merges — entire
function signatures became single tokens because there wasn't enough
pattern diversity to stop greedy merging. This is the same underlying
"ran out of signal" issue seen in the original 5-word toy example,
just more dramatic at this scale.

**Goal (lower priority, mostly a documentation/awareness item):** when
writing up results, explicitly show this failure mode as a deliberate
demonstration of why production tokenizers need large, diverse training
corpora relative to their target vocabulary size — not just a bug to silently
fix.

## 4. Multi-source training (train on many files without holding them all in memory) — ✅ DONE

**Motivation:** a tokenizer trained on a single source (one book, one small
code sample) overfits to that source's quirks, as seen directly in Section 5
(runaway merging on a tiny, repetitive code sample). A real tokenizer needs
many diverse sources — many books, many authors, many codebases — to learn
genuinely generalizable patterns. `train_bpe` currently only accepts one
`text` argument, so this needs new plumbing, not just more input data.

**What was built:**
- `merge_freq_dicts(master, new, stats=None)` — merges a new `word_freqs`
  dictionary into an existing master **in place** (no new dictionary created,
  no return value — follows Python's mutation convention of returning `None`).
  Includes `raise TypeError` validation on both inputs so a user calling this
  directly with wrong types gets an immediate, informative error rather than
  silent data corruption. Optional `stats` parameter accumulates new/old word
  counts for discovery-rate tracking without changing the function's core job.
- `train_bpe_multi(file_list, target_vocab_size, stats_interval=None)` —
  processes one file at a time into `word_freqs`, merges into a running
  `master_freqs` as it goes, converts to `symbol_freqs` exactly once at the
  end, then runs the existing merge loop unchanged. Per-file `try/except`
  ensures one bad file is skipped and flagged immediately without losing
  completed work. Optional `stats_interval` parameter enables new-word
  discovery tracking at configurable checkpoints, returning a list of
  `(book_number, new_percentage)` tuples alongside merges and vocab.

**Key design decisions made during implementation:**
- `merge_freq_dicts` modifies `master` in place rather than returning a new
  dictionary — avoids recreating a growing dictionary on every call, which
  would copy all existing entries even when only a small fraction changed
- `words_to_symbols` called once on the fully-merged `master_freqs`, not per
  file — deduplication happens at the word level (smaller, simpler objects)
  before conversion to the larger symbol-tuple form
- `try/except` wraps each individual file, not the entire loop — one bad file
  skips with a warning while completed work in `master_freqs` is preserved
- Scope boundary documented: `train_bpe_multi` assumes pre-cleaned input;
  source-specific cleaning (e.g. Gutenberg header/footer stripping) is a
  separate, out-of-scope concern

**Experiment run (Section 6):** tested on 10 public domain texts from Project
Gutenberg spanning 1,500 years of written English. Key findings:
1. BPE vocabulary is order-independent — both a descending-size run and a
   randomized run produced identical merges and vocab (332 merges, 500 tokens)
2. New-word discovery tapers but not smoothly — driven by source diversity
   relative to accumulated vocabulary, not just position in sequence
3. Complete duplicates produce exactly 0.0 discovery rate (romeo_clean.txt)
4. "Same work, different edition" can mean something significant — First Folio
   transcription contributed ~40% novelty due to systematic 17th-century
   spelling conventions preserved from the 1623 printed text
5. Domain and genre diversity matter as much as chronological distance —
   The City of God (5th-century theology) and A Room with a View (Edwardian
   fiction) both contributed unexpectedly high novelty relative to their
   position, due to vocabulary domain differences rather than era alone

**Open question (deferred, worth discussing):** at what scale does even the
combined frequency dictionary itself become a memory concern, and would that
require a different approach (e.g. periodic on-disk checkpointing)? Not
implemented, but the design reasoning is documented above.

## 5. (Optional stretch) Compare against a real-world tokenizer

Once v2 pre-tokenization is in place, compare our tokenizer's output on a
shared piece of text against an established tokenizer (e.g. GPT-2's, via
Hugging Face) to see how a production-grade vocabulary (trained on billions
of words) differs from ours.
