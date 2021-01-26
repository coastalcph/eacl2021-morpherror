from glob import glob
from logzero import logger as log
import os
import pyconll
from ufal.udpipe import Model, Pipeline, ProcessingError


def find_best_udpipe_model(model_dir, prefix):
    models = [(f, os.path.getsize(f)) for f in glob(f"{model_dir}/{prefix}*")]
    if not models:
        return None
    return sorted(models, key=lambda t: t[1])[-1][0]


class UDPipe:
    def __init__(self, model_dir, model_name):
        """Instantiates a UDPipe annotator.

        Arguments:
            model_dir: Directory with trained UDPipe models.
            model_name: Name of UDPipe model to use, which can be:
              1. A filename in `model_dir`, in which case this model will
                 simply be loaded.
              2. A language string or prefix (e.g. "english" or "en"),
                 in which case the largest UDPipe model in `model_dir`
                 for that language will be found and loaded.
        """
        if not os.path.isdir(model_dir):
            raise ValueError(f"Not an existing directory: {model_dir}")

        self._modelfile = f"{model_dir}/{model_name}"
        if not os.path.exists(self._modelfile):
            self._modelfile = find_best_udpipe_model(model_dir, model_name)
            if self._modelfile is None or not os.path.exists(self._modelfile):
                raise ValueError(f"No UDPipe model found for name: {model_name}")

        log.info(f"Loading UDPipe model: {self._modelfile}")
        self.model = Model.load(self._modelfile)

    def __call__(self, lines, input_format="horizontal"):
        pipeline = Pipeline(
            self.model, input_format, Pipeline.DEFAULT, Pipeline.NONE, "conllu"
        )
        pe = ProcessingError()

        output = pipeline.process(lines, pe)
        if pe.occurred():
            raise Exception(pe.message)

        return pyconll.load_from_string(output)
