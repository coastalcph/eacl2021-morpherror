from logzero import logger as log
import os
import pyconll
from pycountry import languages
import yaml

from .features import tag_string_features
from .udpipe import UDPipe
from .lexicon import UDLexicon, Unimorph


SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
with open(f"{SCRIPTDIR}/../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
    MODELDIR = config.get("model_dir", "")
    UDLEX_DIR = config.get("udlex_dir", "")


def make_udpipe_language(name):
    if name.startswith("modern greek"):
        return "greek"
    return name


class FeaturePipeline:
    def __init__(
        self,
        language,
        udpipe=True,
        udpipe_modeldir=MODELDIR,
        lexicon="udlex|unimorph",
        udlex_modeldir=UDLEX_DIR,
        string_features=True,
    ):
        """A flexible feature extraction pipeline."""
        self.string_features = string_features

        self.language = None
        try:
            self.language = languages.lookup(language)
        except pycountry.LookupError as e:
            log.exception(e)

        self.udpipe = None
        if udpipe:
            try:
                model_name = make_udpipe_language(self.language.name.lower())
                self.udpipe = UDPipe(udpipe_modeldir, model_name)
            except ValueError as e:
                log.error(e)

        self.lexicon = None
        if lexicon:
            lexica = lexicon.split("|")
            while self.lexicon is None and lexica:
                lexicon = lexica.pop(0)
                if lexicon == "unimorph":
                    try:
                        self.lexicon = Unimorph(self.language.alpha_3)
                        log.info("Using Unimorph")
                    except ValueError as e:
                        log.error(e)
                elif lexicon.startswith("udlex"):
                    try:
                        self.lexicon = UDLexicon(udlex_modeldir, self.language.name)
                        log.info("Using UDLexicon")
                    except ValueError as e:
                        log.error(e)

    def __call__(self, input_):
        """Run pipeline."""
        if isinstance(input_, str):
            if self.udpipe is None:
                raise ValueError(f"Can't annotate string input without UDPipe model.")
            log.debug("Running UDPipe model...")
            input_ = self.udpipe(input_)
        elif self.udpipe is not None:
            if isinstance(input_, pyconll.unit.conll.Conll):
                log.debug("Running UDPipe model on CoNLL input...")
                input_ = self.udpipe(input_.conll(), input_format="conllu")

        if isinstance(input_, pyconll.unit.conll.Conll):
            if self.lexicon is not None:
                log.debug("Running lexical feature extractor...")
                self.lexicon.tag_conll(input_)

            if self.string_features:
                log.debug("Extracting string features...")
                tag_string_features(input_)

        if not isinstance(input_, pyconll.unit.conll.Conll):
            log.error(f"Unknown input format or sth went wrong: {type(input_)}")

        log.debug("Pipeline finished.")

        return input_

    def is_lang(self, language):
        try:
            return self.language == languages.lookup(language)
        except pycountry.LookupError:
            return False
