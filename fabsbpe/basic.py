from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        NUM_BYTES = 256
        num_merges = vocab_size - NUM_BYTES
        ids = list(text.encode("utf-8"))
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(NUM_BYTES)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            new_id = NUM_BYTES + i
            ids = merge(ids, pair, new_id)
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {new_id} ({vocab[new_id]}) had {stats[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            pairs = get_stats(tokens)
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            id = self.merges[pair]
            tokens = merge(tokens, pair, id)
        return tokens

    def decode(self, ids: list[int]):
        tokens = b"".join(self.vocab[id] for id in ids)
        return tokens.decode("utf-8", errors="replace")
