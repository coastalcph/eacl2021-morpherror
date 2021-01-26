from collections import defaultdict
from glob import glob
from logzero import logger as log
import os

from . import Lexicon
from ..conll import ConllFormat


NAME_SUBSTITUTES = {
    "Modern Greek (1453-)": "Greek",
    "Ancient Greek (to 1453)": "Ancient_Greek",
}


def find_best_udlexicon_model(model_dir, prefix):
    if prefix in NAME_SUBSTITUTES:
        prefix = NAME_SUBSTITUTES[prefix]
    models = [
        (f, os.path.getsize(f))
        for f in glob(f"{model_dir}/UDLex_{prefix.capitalize()}*")
    ]
    if not models:
        return None
    return sorted(models, key=lambda t: t[1])[-1][0]


class UDLexicon(Lexicon):
    def __init__(self, model_dir, model_name):
        """Instantiates a UDLexicon.

        Arguments:
            model_dir: Directory with UDLexicon files.
            model_name: Name of UDLexicon file to use, which can be:
              1. A filename in `model_dir`, in which case this model will
                 simply be loaded.
              2. A language string or prefix (e.g. "english" or "en"),
                 in which case the largest UDLexicon in `model_dir`
                 for that language will be found and loaded.
        """
        super().__init__()

        if not os.path.isdir(model_dir):
            raise ValueError(f"Not an existing directory: {model_dir}")

        self._modelfile = f"{model_dir}/{model_name}"
        if not os.path.exists(self._modelfile):
            self._modelfile = find_best_udlexicon_model(model_dir, model_name)
            if self._modelfile is None or not os.path.exists(self._modelfile):
                raise ValueError(f"No UDLexicon file found for name: {model_name}")

        self._load_modelfile()

    def _load_modelfile(self):
        dataset = ConllFormat("UL", discard_errors=True).load_from_file(self._modelfile)
        if dataset.discarded_errors:
            log.warning(f"While loading UDLexicon: discarded {dataset.discarded_errors} erroneous lines")

        all_forms = defaultdict(set)
        all_pos = defaultdict(set)
        self._formcount = {}
        self._lemmacount = {}
        self._lemma = defaultdict(set)

        Col = dataset.columns
        for s in dataset:
            for entry in s:
                form = entry[Col.FORM]
                if "-" in form:
                    continue
                pos = ""
                if entry[Col.UPOS] not in ("", "_"):
                    pos = entry[Col.UPOS]
                    all_pos[form].add(pos)
                if entry[Col.FEATS] not in ("", "_"):
                    feats = entry[Col.FEATS]
                    if pos:
                        feats = f"POS={pos}|{feats}"
                    all_forms[form].add(feats)
                if entry[Col.LEMMA] not in ("", "_"):
                    self._lemma[form].add(entry[Col.LEMMA])

        for entry, forms in all_forms.items():
            self._formcount[entry] = len(forms)
        del all_forms

        for entry, pos in all_pos.items():
            self._poscount[entry] = len(pos)
        del all_pos

        for entry, lemmata in self._lemma.items():
            self._lemmacount[entry] = len(lemmata)

    def get_lemma(self, form):
        num_lemma = self._lemmacount.get(form, 0)
        if num_lemma == 0:
            form = form.lower()
            num_lemma = self._lemmacount.get(form, 0)
        if num_lemma == 0:
            form = form.capitalize()
            num_lemma = self._lemmacount.get(form, 0)
        if num_lemma == 0:
            return None

        return self._lemma[form].copy().pop()
