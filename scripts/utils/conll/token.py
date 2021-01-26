from enum import IntEnum


def _field_to_str(field):
    if not field:
        return "_"
    return str(field)


class Token:
    def __init__(self, fields=None):
        self._fields = []
        if fields is not None:
            self._fields = fields

    def __bool__(self):
        return len(self._fields) > 0

    def __getitem__(self, idx):
        return self._fields[idx]

    def __setitem__(self, idx, item):
        self._fields[idx] = item

    def __eq__(self, obj):
        if isinstance(obj, Token):
            return self._fields == obj._fields
        return self._fields == list(obj)

    @property
    def is_multiword(self):
        # assumes the first column is an ID column
        return "-" in self._fields[0]

    def to_string(self, transform=None):
        fields = self._fields
        if transform is not None:
            fields = transform(fields)
        return "\t".join(_field_to_str(f) for f in fields)
