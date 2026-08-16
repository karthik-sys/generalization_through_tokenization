"""SOTA-tokenizer comparison arm: tiktoken's cl100k_base (GPT-4/3.5's real production
tokenizer), used with the same tiny backbone as MoT and our own trained BPE baseline.

This is a genuine third arm, not a like-for-like param-matched one: cl100k_base's real
vocab is 100,277 tokens, which is accepted as-is rather than artificially shrunk - the
whole point is testing against a real industrial-grade tokenizer, not a scaled-down
imitation of one. Its embedding+head will dwarf the other two arms' (~25.7M params vs
MoT's 8.1M / our BPE baseline's 4.4M); that asymmetry is inherent to what "SOTA
tokenizer, matched backbone" means, not a bug to fix.
"""

import tiktoken


class SotaTokenizer:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.enc = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special=set())

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)
