from utils import get_stats, merge

class BasicTokenizer:
    
    def __init__(self):
        self.merges = {}
        self.vocab = {}
    
    def train(self, text, vocab_size, verbose=False):
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
        num_merges = vocab_size - 256
        ids = list(tokens)

        self.merges = {}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            if verbose:
                print(f"merging {pair} into a new token {idx}")
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
    
    def encode(self, text):
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience

        while True:
            s = get_stats(tokens)
            pair = min(s, key=lambda x: self.merges.get(x, float('inf')))
            if pair not in self.merges:
                break
            tokens = merge(tokens, pair, self.merges[pair])

        return tokens
    
    def decode(self, ids):
        original = b''
        for _id in ids:
            original += self.vocab[_id]
        return original.decode('utf-8', errors="replace")
    