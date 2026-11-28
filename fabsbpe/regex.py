
from fabsbpe.base import Tokenizer, get_stats, merge

import regex as re

class RegexTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        NUM_BYTES = 256
        num_merges = vocab_size - NUM_BYTES
        chunks = re.findall(self.GPT4_SPLIT_PATTERN, text)
        chunks_ids = list(map(lambda c: list(c.encode("utf-8")), chunks))
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(NUM_BYTES)}
        for i in range(num_merges):
            stats = {}
            new_chunk_ids = []
            pair = None
            for ids in chunks_ids:
                get_stats(ids, stats)
            pair = max(stats, key=stats.get)
            new_id = NUM_BYTES + i
            for ids in chunks_ids:
                new_chunk_ids.append(merge(ids, pair, new_id))
            chunks_ids = new_chunk_ids
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {new_id} ({vocab[new_id]}) had {stats[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab
    def encode(self, text: str):
        chunks = re.findall(self.GPT4_SPLIT_PATTERN, text)
        chunks_id = list(map(lambda c: c.encode("utf-8"), chunks))
        tokens = []
        for chunk in chunks_id:
            while len(chunk) >= 2:
                pairs = get_stats(chunk)
                pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                id = self.merges[pair]
                chunk = merge(chunk, pair, id)
            tokens.extend(chunk)
        return tokens
    def decode(self, ids: list[int]):
        tokens = b"".join(self.vocab[id] for id in ids)
        return tokens.decode("utf-8", errors="replace")

            