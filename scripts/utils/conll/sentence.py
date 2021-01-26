class Sentence:
    def __init__(self):
        self._tokens = []
        self._metadata = []

    def append_metadata(self, data):
        self._metadata.append(data)

    def append_token(self, token):
        self._tokens.append(token)

    def __bool__(self):
        return len(self._tokens) > 0

    def __len__(self):
        return len(self._tokens)

    def __iter__(self):
        for tok in self._tokens:
            yield tok

    def __getitem__(self, idx):
        return self._tokens[idx]

    @property
    def metadata(self):
        return self._metadata

    def to_string(self, col_transform=None):
        s = [f"# {m}" for m in self._metadata]
        s += [t.to_string(transform=col_transform) for t in self._tokens]
        return "\n".join(s)
