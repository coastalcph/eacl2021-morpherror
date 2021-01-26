from bidict import bidict
import edlib


def ascii_encode_with_map(cmap, s):
    for a, b in cmap.items():
        s = s.replace(a, b)
    return s


def ascii_encode(*args):
    codes = set(range(1, 128))
    cmap = bidict()
    alphabet = set(c for word in args for c in word)
    for char in alphabet:
        if ord(char) < 128:
            codes.remove(ord(char))
    for char in alphabet:
        if ord(char) >= 128:
            cmap[char] = chr(codes.pop())
    if not cmap:
        return args
    return [ascii_encode_with_map(cmap, arg) for arg in args]


def align(query, target, mode="NW", task="distance", **kwargs):
    enc_query, enc_target = ascii_encode(query, target)
    kwargs["mode"] = mode
    kwargs["task"] = task
    return edlib.align(enc_query, enc_target, **kwargs)


def distance(self, query, target, **kwargs):
    if query == target:
        return 0
    return align(query, target, **kwargs).get("editDistance")


def nice_align(query, target, **kwargs):
    aligned = align(query, target, task="path", **kwargs)
    return edlib.getNiceAlignment(aligned, query, target)["matched_aligned"]
