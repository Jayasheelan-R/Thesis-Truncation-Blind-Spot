"""Rare-token / contamination metrics measuring how a generation's data
compares to a fixed human-text baseline across recursive retraining
generations. All six metrics are computed with plain numpy (constitution
Principle V: no new dependency needed -- see research.md item 4).
"""
import json
import os
import re
from datetime import datetime, timezone

import numpy as np

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize_words(texts):
    """Lowercase word tokenization used consistently for every metric below."""
    words = []
    for text in texts:
        words.extend(w.lower() for w in _WORD_RE.findall(text))
    return words


def word_frequencies(words):
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq


def hapax_rate(freq):
    """Fraction of unique word types occurring exactly once."""
    if not freq:
        return 0.0
    hapax_count = sum(1 for c in freq.values() if c == 1)
    return hapax_count / len(freq)


def zipf_exponent(freq):
    """Negative slope of log(frequency) vs log(rank), via linear regression."""
    if len(freq) < 2:
        return 0.0
    counts = np.array(sorted(freq.values(), reverse=True), dtype=np.float64)
    ranks = np.arange(1, len(counts) + 1, dtype=np.float64)
    log_ranks = np.log(ranks)
    log_counts = np.log(counts)
    slope, _intercept = np.polyfit(log_ranks, log_counts, 1)
    return float(-slope)


def survival_rate(baseline_word_types, current_freq):
    """Fraction of baseline_word_types still present (any frequency) in current_freq."""
    if not baseline_word_types:
        return 0.0
    present = sum(1 for w in baseline_word_types if w in current_freq)
    return present / len(baseline_word_types)


def unigram_entropy(freq):
    """Shannon entropy, in bits, of the unigram frequency distribution."""
    if not freq:
        return 0.0
    counts = np.array(list(freq.values()), dtype=np.float64)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def kl_divergence_from_baseline(baseline_freq, current_freq, epsilon=1e-10):
    """KL divergence of current_freq's distribution from baseline_freq's,
    over the union of both vocabularies, with additive smoothing."""
    vocab = set(baseline_freq) | set(current_freq)
    if not vocab:
        return 0.0

    baseline_total = sum(baseline_freq.values()) + epsilon * len(vocab)
    current_total = sum(current_freq.values()) + epsilon * len(vocab)

    kl = 0.0
    for w in vocab:
        p = (current_freq.get(w, 0) + epsilon) / current_total
        q = (baseline_freq.get(w, 0) + epsilon) / baseline_total
        kl += p * np.log(p / q)
    return float(kl)


def rare_decile_word_types(freq):
    """The bottom decile (rarest 10%) of word types by frequency."""
    if not freq:
        return []
    sorted_words = sorted(freq.items(), key=lambda kv: kv[1])
    n_decile = max(1, len(sorted_words) // 10)
    return [w for w, _c in sorted_words[:n_decile]]


def get_or_create_baseline(experiment_path, human_texts):
    """Load baseline_rare_tokens.json if it already exists (run-level, not
    per-iteration); otherwise compute it once from human_texts and write it.
    Never recomputed once it exists (spec FR-004/SC-004)."""
    baseline_path = os.path.join(str(experiment_path), "baseline_rare_tokens.json")

    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            return json.load(f)

    words = tokenize_words(human_texts)
    freq = word_frequencies(words)

    baseline = {
        "hapax_word_types": [w for w, c in freq.items() if c == 1],
        "rare_decile_word_types": rare_decile_word_types(freq),
        "unigram_frequencies": freq,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(str(experiment_path), exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=4)

    return baseline


def compute_contamination_metrics(generation_texts, baseline):
    """Compute all six metrics for one generation's texts against the
    already-established baseline record (from get_or_create_baseline)."""
    words = tokenize_words(generation_texts)
    freq = word_frequencies(words)

    return {
        "hapax_rate": hapax_rate(freq),
        "zipf_exponent": zipf_exponent(freq),
        "hapax_survival_rate": survival_rate(baseline["hapax_word_types"], freq),
        "rare_decile_survival_rate": survival_rate(baseline["rare_decile_word_types"], freq),
        "unigram_entropy": unigram_entropy(freq),
        "kl_divergence_from_human_baseline": kl_divergence_from_baseline(baseline["unigram_frequencies"], freq),
    }
