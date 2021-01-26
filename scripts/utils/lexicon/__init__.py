from functools import lru_cache


class Lexicon:
    def __init__(self):
        self._formcount = {}
        self._lemmacount = {}
        self._poscount = {}

    def get_lemma(self, form):
        return None

    @lru_cache(maxsize=4096)
    def get_tags(self, form):
        tags = set()
        if not form:
            return tags

        num_entries = self._formcount.get(form, 0)
        if num_entries == 0:
            form = form.lower()
            num_entries = self._formcount.get(form, 0)
        if num_entries == 0:
            form = form.capitalize()
            num_entries = self._formcount.get(form, 0)
        if num_entries == 0:
            return tags

        if num_entries > 4:
            tags.add("^syncretic")
        elif num_entries > 1:
            tags.add("+syncretic")
        elif num_entries == 1:
            tags.add("-syncretic")

        if form in self._poscount:
            num_pos = self._poscount.get(form, 0)
            if num_pos > 4:
                tags.add("^posambig")
            elif num_pos > 1:
                tags.add("+posambig")
            elif num_pos == 1:
                tags.add("-posambig")

        if form in self._lemmacount:
            if self._lemmacount.get(form) > 1:
                tags.add("+lexambig")
            elif self._lemmacount.get(form) == 1:
                tags.add("-lexambig")

        return tags

    def tag_conll(self, conll):
        for sent in conll:
            for token in sent:
                if token.lemma is None:
                    token.lemma = self.get_lemma(token.form)
                tags = self.get_tags(token.form)
                if tags:
                    token.misc["lex"] = tags


from .udlexicon import UDLexicon
from .unimorph import Unimorph
