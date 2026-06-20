# BPE Tokenizer — Version 2 Goals

Captured after building and testing the v1 tokenizer on:
- A 5-word toy example (hand-traced)
- Alice's Adventures in Wonderland (full book, target_vocab_size=500)
- A short Python code sample (3 functions, target_vocab_size=150)

## 1. Pre-tokenization: split punctuation from words before BPE runs

**Problem observed:** our current `get_word_freqs` only splits on whitespace
(`text.split()`), so punctuation stays glued to words.

Evidence from Alice in Wonderland training:
- `('i', 's,_')` — "is," and "is" are treated as unrelated tokens
- `('k', ',_')`, `('h', ',_')` — commas fused onto word-ending letters

Evidence from Python code training (more severe):
- `('get_pair_freqs(symbol_freqs', '):_')` — an entire function signature
  collapsed into one token
- `('tuple(list(word', ')_')` — parens and method calls fused into identifiers

**Goal:** write a pre-tokenization step that separates punctuation
(`. , ( ) [ ] { } : ; = + - " '` etc.) into their own standalone units
*before* character-level BPE training begins. This should let BPE learn
cleaner, more generalizable subword patterns instead of memorizing
punctuation-glued artifacts.

**Open question to explore:** should the apostrophe in contractions/possessives
("Alice's", "don't") be treated as a special case and kept attached, since it
carries meaning — vs. periods/commas/brackets, which are purely structural?

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

## 4. (Optional stretch) Compare against a real-world tokenizer

Once v2 pre-tokenization is in place, compare our tokenizer's output on a
shared piece of text against an established tokenizer (e.g. GPT-2's, via
Hugging Face) to see how a production-grade vocabulary (trained on billions
of words) differs from ours.
