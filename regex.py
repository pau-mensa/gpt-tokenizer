import regex as re
from utils import get_stats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer:
    
    def __init__(self):
        self.merges = {}
        self.pattern = GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
    
    def train(self, text, vocab_size, verbose=False):
        tokens = re.findall(self.compiled_pattern, text)
        sentence = [list(l.encode('utf-8')) for l in tokens]
        num_merges = vocab_size - 256

        self.merges = {}
        for i in range(num_merges):
            stats = {}
            for chunk in sentence:
                n_stats = get_stats(chunk)
                for k, v in n_stats.items():
                    stats[k] = stats.get(k, 0) + n_stats[k]
            pair = max(stats, key=stats.get)
            idx = 256 + i
            if verbose:
                print(f"merging {pair} into a new token {idx}")
            for s_idx, chunk in enumerate(sentence):
                new_ids = merge(chunk, pair, idx)
                sentence[s_idx] = new_ids
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
    