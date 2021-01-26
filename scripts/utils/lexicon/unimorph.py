from logzero import logger as log
import unimorph

from . import Lexicon


def unimorph_substitute(language):
    subst = False

    if language == "nor":
        log.warning(f"Need to specify 'nob' (Bokmål) or 'nno' (Nynorsk) for Norwegian")
    elif language == "glg":
        subst = "gal"
    elif language in ("hrv", "srp"):
        subst = "hbs"
    elif language == "slk":
        subst = "ces"

    if subst:
        log.warning(
            f"Language code '{language}' not in UniMorph, but can substitute with '{subst}'"
        )

    return subst


class Unimorph(Lexicon):
    datasets = None

    def __init__(self, language):
        """Instantiates a Unimorph lexicon.

        Arguments:
            language: Three-letter language code.
        """
        super().__init__()

        if not self.has_dataset(language):
            subst = unimorph_substitute(language)
            if not subst or not self.has_dataset(subst):
                raise ValueError(f"Not a valid language in UniMorph: {language}")
            language = subst

        self._data = unimorph.load_dataset(language)
        self._data["pos"] = self._data["features"].apply(lambda x: x.split(";")[0])
        self._formcount = self._data.groupby("form").nunique()["features"]
        self._lemmacount = self._data.groupby("form").nunique()["lemma"]
        self._poscount = self._data.groupby("form").nunique()["pos"]

    @classmethod
    def has_dataset(cls, language):
        if not cls.datasets:
            cls.datasets = unimorph.get_list_of_datasets()
            if not cls.datasets:
                log.error(
                    "Couldn't get list of Unimorph datasets -- requires internet connection"
                )
                return True
        return language in cls.datasets

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

        lemmata = set(self._data[self._data["form"] == form]["lemma"])
        return lemmata.pop()
