"""SentencePiece BPE tokenizer, used per-domain for code/math/science (spec §6) and for
the unified baseline (spec §9, Baseline A). Byte-fallback keeps every domain robust to OOV.
"""

from pathlib import Path

import sentencepiece as spm


def train_bpe(
    texts: list[str],
    model_prefix: str,
    vocab_size: int,
    model_type: str = "bpe",
) -> None:
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{model_prefix}.train_input.txt"
    Path(tmp_path).write_text("\n".join(texts), encoding="utf-8")

    # vocab_size must be <= number of distinct pieces obtainable from the corpus;
    # small stage-1 corpora (esp. math) can't support a large vocab.
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        byte_fallback=True,
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<|endofdoc|>"],
    )
    Path(tmp_path).unlink()


class BPETokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)
