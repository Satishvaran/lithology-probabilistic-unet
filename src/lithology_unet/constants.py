"""FORCE 2020 labels published with the competition dataset."""

LITHOLOGY_CODE_TO_NAME = {
    30000: "Sandstone",
    65030: "Sandstone/Shale",
    65000: "Shale",
    80000: "Marl",
    74000: "Dolomite",
    70000: "Limestone",
    70032: "Chalk",
    88000: "Halite",
    86000: "Anhydrite",
    99000: "Tuff",
    90000: "Coal",
    93000: "Basement",
}

LITHOLOGY_CODES = tuple(LITHOLOGY_CODE_TO_NAME)
CODE_TO_INDEX = {code: i for i, code in enumerate(LITHOLOGY_CODES)}
INDEX_TO_CODE = {i: code for code, i in CODE_TO_INDEX.items()}
INDEX_TO_NAME = {i: LITHOLOGY_CODE_TO_NAME[c] for i, c in INDEX_TO_CODE.items()}
IGNORE_INDEX = -100
