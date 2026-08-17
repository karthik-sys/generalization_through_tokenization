"""NLP domain tokenizer: surface-subword -> syllable -> morpheme -> byte encoding
backoff (spec §7). v1 default is this single-stream backoff, not parallel multi-stream
fusion - see decision log in docs/architecture_spec.md for why.

Each emitted token carries a (domain_vocab_id, type_id) pair so the model can condition
on granularity as well as identity (spec §6): SURFACE=0, SYLLABLE=1, MORPHEME=2, BYTE=3.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

import morfessor
import pyphen
import sentencepiece as spm

SURFACE, SYLLABLE, MORPHEME, BYTE = 0, 1, 2, 3
NUM_TYPES = 4

WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Reserved ids within each within-type sub-vocab (mirrors SentencePiece's own convention).
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
NUM_SPECIAL = 4


def _words(texts: list[str]) -> list[str]:
    return WORD_RE.findall(" ".join(texts).lower())


def train_nlp_tokenizer(
    texts: list[str],
    out_dir: str,
    surface_vocab_size: int = 4000,
    syllable_vocab_size: int = 4000,
    morpheme_vocab_size: int = 4000,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Surface subwords (SentencePiece unigram - tried first at encode time)
    tmp = out / "surface_train_input.txt"
    tmp.write_text("\n".join(texts), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(tmp),
        model_prefix=str(out / "surface"),
        vocab_size=surface_vocab_size,
        model_type="unigram",
        byte_fallback=True,
        character_coverage=0.9995,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        num_threads=os.cpu_count() or 16,
    )
    tmp.unlink()

    words = _words(texts)
    word_counts = Counter(words)

    # 2. Syllable vocab (rule-based syllabifier, pyphen)
    dic = pyphen.Pyphen(lang="en")
    syllable_counts: Counter = Counter()
    for w, c in word_counts.items():
        for syl in dic.inserted(w).split("-"):
            if syl:
                syllable_counts[syl] += c
    # capped by frequency, same reasoning as any BPE vocab_size cap: an uncapped
    # syllable/morpheme vocab grows with corpus size (long tail of rare segments),
    # which silently balloons the model's embedding/head params - rare segments
    # fall back further down the chain (morpheme, then byte) instead, as intended.
    syllable_vocab = [s for s, _ in syllable_counts.most_common(syllable_vocab_size)]
    (out / "syllable_vocab.json").write_text(json.dumps(syllable_vocab), encoding="utf-8")

    # 3. Morpheme vocab (Morfessor, unsupervised segmentation learned from corpus)
    model = morfessor.BaselineModel()
    model.load_data([(c, w) for w, c in word_counts.items()])
    model.train_batch()
    io = morfessor.MorfessorIO()
    io.write_binary_model_file(str(out / "morfessor.bin"), model)

    morph_counts: Counter = Counter()
    for w, c in word_counts.items():
        for morph in model.viterbi_segment(w)[0]:
            morph_counts[morph] += c
    morpheme_vocab = [m for m, _ in morph_counts.most_common(morpheme_vocab_size)]
    (out / "morpheme_vocab.json").write_text(json.dumps(morpheme_vocab), encoding="utf-8")


class NLPHybridTokenizer:
    """Encode/decode with the surface -> syllable -> morpheme -> byte backoff.

    A "word" backs off past surface when SentencePiece can't represent it as a
    single piece (i.e. it would take >1 surface token or hit UNK) - that's the
    OOV/low-quality signal spec §7 calls for. Non-word characters (punctuation,
    whitespace) always go through the surface tokenizer, which byte-falls-back itself.
    """

    def __init__(self, model_dir: str):
        d = Path(model_dir)
        self.surface = spm.SentencePieceProcessor(model_file=str(d / "surface.model"))
        self.syllable_vocab = json.loads((d / "syllable_vocab.json").read_text())
        self.syllable_id = {s: i for i, s in enumerate(self.syllable_vocab)}
        self.morpheme_vocab = json.loads((d / "morpheme_vocab.json").read_text())
        self.morpheme_id = {m: i for i, m in enumerate(self.morpheme_vocab)}
        io = morfessor.MorfessorIO()
        self.morfessor_model = io.read_binary_model_file(str(d / "morfessor.bin"))
        self.dic = pyphen.Pyphen(lang="en")

    @property
    def vocab_sizes(self) -> dict[str, int]:
        return {
            "surface": self.surface.vocab_size(),
            "syllable": len(self.syllable_vocab) + NUM_SPECIAL,
            "morpheme": len(self.morpheme_vocab) + NUM_SPECIAL,
            "byte": 256 + NUM_SPECIAL,
        }

    def _encode_word(self, word: str) -> list[tuple[int, int]]:
        pieces = self.surface.encode(word, out_type=int)
        if len(pieces) == 1 and pieces[0] != UNK_ID:
            return [(pieces[0], SURFACE)]

        syllables = [s for s in self.dic.inserted(word).split("-") if s]
        if syllables and all(s in self.syllable_id for s in syllables):
            return [(self.syllable_id[s] + NUM_SPECIAL, SYLLABLE) for s in syllables]

        morphemes = self.morfessor_model.viterbi_segment(word)[0]
        if morphemes and all(m in self.morpheme_id for m in morphemes):
            return [(self.morpheme_id[m] + NUM_SPECIAL, MORPHEME) for m in morphemes]

        return [(b + NUM_SPECIAL, BYTE) for b in word.encode("utf-8")]

    def encode(self, text: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        pos = 0
        for m in WORD_RE.finditer(text):
            # preserve inter-token whitespace as surface tokens too, for round-trip fidelity
            if m.start() > pos:
                gap = text[pos : m.start()]
                out.extend((p, SURFACE) for p in self.surface.encode(gap, out_type=int))
            token = m.group()
            if token.isalpha() or (token.isalnum() and not token.isdigit()):
                out.extend(self._encode_word(token.lower()))
            else:
                out.extend((p, SURFACE) for p in self.surface.encode(token, out_type=int))
            pos = m.end()
        if pos < len(text):
            out.extend((p, SURFACE) for p in self.surface.encode(text[pos:], out_type=int))
        return out
