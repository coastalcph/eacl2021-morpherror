from enum import IntEnum
import gzip

from .sentence import Sentence
from .token import Token


class ConllParseError(Exception):
    def __init__(self, message, line, line_no):
        super().__init__(message)
        self.message = message
        self.line = line
        self.line_no = line_no

    def __str__(self):
        return f"on line {self.line_no}: {self.message}\nLine was: {self.line}"


class ConllDataset:
    def __init__(self, columns=None, last_column_multi=False, discard_errors=False):
        self.last_column_multi = last_column_multi
        self.columns = columns
        self.discard_errors = discard_errors
        self.discarded_errors = 0
        self._sentences = []

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, cols):
        if cols is None:
            self._columns = None
            self._colcount = 0
        else:
            self._columns = IntEnum("Column", cols, start=0)
            self._colcount = len(cols)

    def __len__(self):
        return len(self._sentences)

    def __iter__(self):
        for s in self._sentences:
            yield s

    def __getitem__(self, idx):
        return self._sentences[idx]

    def load(self, f):
        s = Sentence()
        self.discarded_errors = 0
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                if s:
                    self._sentences.append(s)
                    s = Sentence()
            elif line.startswith("# "):
                s.append_metadata(line[2:])
            else:
                try:
                    tok = self._parse_token(line)
                except ValueError as e:
                    if self.discard_errors:
                        self.discarded_errors += 1
                        continue
                    raise ConllParseError(str(e), line, i + 1) from None
                s.append_token(tok)
        if s:
            self._sentences.append(s)
        return self

    def load_from_file(self, filename):
        if filename.endswith(".gz"):
            with gzip.open(filename, "rt") as f:
                self.load(f)
        else:
            with open(filename, "r") as f:
                self.load(f)
        return self

    def matches_shape(self, obj):
        """Returns True if the shape of this dataset matches `obj`.

        Compares sentence lengths to check if both datasets could represent
        annotations of the same underlying data, but without actually comparing
        FORMs."""
        return (len(self) == len(obj)) and all(
            len(a) == len(b) for a, b in zip(self, obj)
        )

    def as_horizontal_text(self, idx=None):
        if idx is None:
            idx = self.columns["FORM"]
        return "\n".join(" ".join(tok[idx] for tok in s) for s in self)

    def to_string(self, col_transform=None):
        return "\n\n".join(
            s.to_string(col_transform=col_transform) for s in self._sentences
        )

    def _parse_token(self, line):
        columns = line.split("\t")
        if not (
            len(columns) == self._colcount
            or (self.last_column_multi and len(columns) >= self._colcount - 1)
        ):
            msg = f"Line has {len(columns)} columns, but expected {self._colcount}"
            if self.last_column_multi:
                msg += " or more"
            raise ValueError(msg)

        if self.last_column_multi:
            columns = columns[: self._colcount - 1] + [columns[self._colcount - 1 :]]

        return Token(columns)
