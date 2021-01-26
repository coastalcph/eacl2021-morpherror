import pyconll

from .dataset import ConllDataset
from .formats import ConllFormat


def pyconll_load(filename):
    try:
        conll = pyconll.load_from_file(filename)
    except pyconll.exception.ParseError as e:
        log.warning(f"pyconll.ParseError: {e}")
        log.warning("-- attempting empty fields fix...")

        with open(filename, "r") as f:
            lines = f.read()

        lines = lines.replace("\t\n", "\t_\n")
        while "\t\t" in lines:
            lines = lines.replace("\t\t", "\t_\t")

        try:
            conll = pyconll.load_from_string(lines)
        except pyconll.exception.ParseError as e:
            log.error(f"pyconll.ParseError: {e}")
            return None

    return conll
