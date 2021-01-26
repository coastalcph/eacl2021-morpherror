from .dataset import ConllDataset


CONLL2009_COLUMNS = [
    "ID",
    "FORM",
    "LEMMA",
    "PLEMMA",
    "POS",
    "PPOS",
    "FEAT",
    "PFEAT",
    "HEAD",
    "PHEAD",
    "DEPREL",
    "PDEPREL",
    "FILLPRED",
    "PRED",
    "APRED",
]

CONLL_U_COLUMNS = [
    "ID",
    "FORM",
    "LEMMA",
    "UPOS",
    "XPOS",
    "FEATS",
    "HEAD",
    "DEPREL",
    "DEPS",
    "MISC",
]

PARSEME_COLUMNS = CONLL_U_COLUMNS + ["MWE"]

CONLL_UL_COLUMNS = [
    "FROM",
    "TO",
    "FORM",
    "LEMMA",
    "UPOS",
    "CPOS",
    "FEATS",
    "MISC",
    "ANCHORS",
]


def ConllFormat(fmt, **kwargs):
    if fmt == "2009" or fmt == 2009:
        return ConllDataset(columns=CONLL2009_COLUMNS, last_column_multi=True, **kwargs)
    elif fmt.lower() == "u":
        return ConllDataset(columns=CONLL_U_COLUMNS, **kwargs)
    elif fmt.lower() == "ul":
        return ConllDataset(columns=CONLL_UL_COLUMNS, last_column_multi=True, **kwargs)
    elif fmt.lower() == "parseme":
        return ConllDataset(columns=PARSEME_COLUMNS, **kwargs)

    raise ValueError(f"Unknown ConllFormat: {fmt}")
