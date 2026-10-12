"""Best known side lengths s(n) for packing n unit squares in a square.

Values copied from the "Squares in Squares" table of Erich Friedman's Packing
Center, now maintained by David Ellsworth:
https://kingbird.myphotos.cc/packing/squares_in_squares.html (checked 2026-10-12).

That page states that for every n <= 324 it does not picture, the trivial
axis-aligned packing (side ceil(sqrt(n))) is the best known. Only n <= 55 are
transcribed here; for larger n, ``best_known`` returns None.
"""

import math

_SQRT2 = math.sqrt(2.0)
_SQRT7 = math.sqrt(7.0)

# n -> (side, note). n missing from this dict (and <= 55) use ceil(sqrt(n)).
_NONTRIVIAL = {
    5: (2 + _SQRT2 / 2, "2 + sqrt(2)/2, proven optimal (Göbel 1979)"),
    10: (3 + _SQRT2 / 2, "3 + sqrt(2)/2, proven optimal (Stromquist 2003)"),
    11: (3.87708359002281, "root of a degree-8 polynomial"),
    17: (4.67553009360455, "root of a degree-18 polynomial"),
    18: (3.5 + _SQRT7 / 2, "7/2 + sqrt(7)/2"),
    19: (3 + 4 * _SQRT2 / 3, "3 + 4 sqrt(2)/3"),
    26: (3.5 + 1.5 * _SQRT2, "7/2 + 3 sqrt(2)/2"),
    27: (5 + _SQRT2 / 2, "5 + sqrt(2)/2"),
    28: (5.82444461667405, "root of a degree-6 polynomial"),
    29: (5.93383346267692, "numerical"),
    37: (6.59861960924436, "root of a degree-8 polynomial"),
    38: (6 + _SQRT2 / 2, "6 + sqrt(2)/2"),
    39: (6.81072208306864, "root of a degree-5 polynomial"),
    40: (4 + 2 * _SQRT2, "4 + 2 sqrt(2)"),
    41: (6.92669309446880, "root of a degree-42 polynomial"),
    50: (7 + 4 / 7, "7 + 4/7"),
    51: (7.70079923541701, "root of a degree-12 polynomial"),
    52: (7 + _SQRT2 / 2, "7 + sqrt(2)/2"),
    53: (6.5 + _SQRT7 / 2, "13/2 + sqrt(7)/2"),
    54: (7 - _SQRT2 / 2 + math.sqrt(1 + _SQRT2), "7 - sqrt(2)/2 + sqrt(1 + sqrt(2))"),
    55: (7.94577100750391, "numerical"),
}
MAX_TABULATED = 55


def best_known(n):
    """Best known side for n squares, or None if not tabulated here."""
    if n < 1:
        raise ValueError("n must be positive")
    if n in _NONTRIVIAL:
        return _NONTRIVIAL[n][0]
    if n <= MAX_TABULATED:
        return float(math.ceil(math.sqrt(n) - 1e-12))
    return None


def describe(n):
    """Human-readable provenance of the best known value for n."""
    if n in _NONTRIVIAL:
        return _NONTRIVIAL[n][1]
    if n <= MAX_TABULATED:
        return "trivial axis-aligned packing"
    return "not tabulated"


def lower_bound(n):
    """Trivial area bound: n unit squares need a container of side >= sqrt(n)."""
    return math.sqrt(n)


def trivial_upper_bound(n):
    """Side of the plain grid packing, ceil(sqrt(n))."""
    return float(math.ceil(math.sqrt(n) - 1e-12))
