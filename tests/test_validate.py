import math

import numpy as np

from sqpack.validate import to_container_frame, validate


def test_grid_packing_is_valid():
    cfg = np.array([[x + 0.5, y + 0.5, 0.0] for x in range(3) for y in range(3)])
    v = validate(cfg)
    assert v.valid and abs(v.side - 3) < 1e-12


def test_overlap_is_detected():
    cfg = np.array([[0, 0, 0], [0.7, 0.2, 0.4]], dtype=float)
    v = validate(cfg)
    assert not v.valid and v.max_overlap > 0.1 and v.worst_pair == (0, 1)


def test_known_optimum_for_five_squares():
    # four squares in the corners of a (2 + sqrt2/2)-box, one at 45 degrees in the middle
    s = 2 + math.sqrt(2) / 2
    cfg = np.array([[0.5, 0.5, 0], [s - 0.5, 0.5, 0], [0.5, s - 0.5, 0], [s - 0.5, s - 0.5, 0], [s / 2, s / 2, math.pi / 4]])
    v = validate(cfg)
    assert v.valid
    assert abs(v.side - s) < 1e-12


def test_container_frame_starts_at_origin():
    cfg = np.array([[10.0, -4.0, 0.0], [11.0, -4.0, 0.0]])
    out, s = to_container_frame(cfg)
    assert abs(s - 2) < 1e-12
    xs = np.concatenate([out[:, 0] - 0.5, out[:, 0] + 0.5])
    assert abs(xs.min()) < 1e-12 and abs(xs.max() - 2) < 1e-12
