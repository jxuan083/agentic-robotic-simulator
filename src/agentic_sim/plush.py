"""Finals plush toy (a Shiba Inu sitting in an Enoden 305 tram): dimensions, mass and colours in one place.

Source: notes/261128_玩偶規格與決賽用法.md (11/28). Length, width, body height, total height and mass are the user's
measurements; the head size, paws and colours are read from the photos (head width still to be measured). The
simulation models the fabric toy as rigid primitives (no soft body); the real side can import the same numbers
(detection thresholds, slot sizes).

Plush frame: origin at the centre of the bottom face, +X along the tram (90 mm), +Y across (60 mm) towards the
dog's face, +Z up.
"""

from __future__ import annotations

ID = "plush"
NAME_ZH = "柴犬玩偶"

LENGTH = 0.090        # tram length, measured
WIDTH = 0.060         # tram width, measured
BODY_HEIGHT = 0.050   # tram body height, measured
TOTAL_HEIGHT = 0.100  # with the dog's head, measured
MASS = 0.0286         # without the bead chain and tag (finals state), measured
HEAD_MASS = 0.0050    # estimate: head about a fifth of the stuffed volume, so the centre of mass stays in the body
CORNER_RADIUS = 0.010  # rounded vertical tram edges (photos)

# Head: an ellipsoid on the roof, centred along the tram; top at TOTAL_HEIGHT. Radii (x, y, z) from the photos
# (50-55 mm wide, to be measured).
HEAD_RADII = (0.025, 0.026, 0.025)
HEAD_POS = (0.0, 0.0, TOTAL_HEIGHT - HEAD_RADII[2])
MUZZLE_RADII = (0.012, 0.008, 0.009)    # off-white muzzle on the +Y side of the head (visual only)
PAW_RADIUS = 0.008                      # two front paws on the roof edge, +Y side

FRICTION = 1.0   # fabric; contacts with the gripper use the gripper's own settings (Menagerie priority 1)

# sRGB colours read from the photos
COLOR_BODY = "#1E6B4A"     # tram upper and lower body, roof (dark green)
COLOR_BAND = "#E8E39A"     # waist band (pale yellow)
COLOR_WINDOW = "#8FC3E6"   # windows (light blue)
COLOR_UNDER = "#9A9A9A"    # underside (grey)
COLOR_DOG = "#C8955A"      # dog (tan)
COLOR_FACE = "#F2EDE4"     # face, muzzle (off-white)

# Visual bands on the tram sides (height ranges above the bottom face); the windows sit inside the band
UNDER_BAND = (0.0, 0.006)
WAIST_BAND = (0.016, 0.038)
WINDOW_BAND = (0.020, 0.034)


def rgb(hex_color: str) -> tuple:
    """'#RRGGBB' -> (r, g, b) in 0..1 (sRGB, as written in the scene files)."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def inertia() -> tuple:
    """(centre of mass, principal moments) in the plush frame: a solid tram box plus a solid head ellipsoid."""
    mb, mh = MASS - HEAD_MASS, HEAD_MASS
    zb, zh = BODY_HEIGHT / 2, HEAD_POS[2]
    zc = (mb * zb + mh * zh) / MASS
    L, W, H = LENGTH, WIDTH, BODY_HEIGHT
    a, b, c = HEAD_RADII
    ib = (mb / 12 * (W * W + H * H), mb / 12 * (L * L + H * H), mb / 12 * (L * L + W * W))
    ih = (mh / 5 * (b * b + c * c), mh / 5 * (a * a + c * c), mh / 5 * (a * a + b * b))
    shift = mb * (zb - zc) ** 2 + mh * (zh - zc) ** 2   # parallel axis, x and y only (both centres on the z axis)
    return (0.0, 0.0, zc), (ib[0] + ih[0] + shift, ib[1] + ih[1] + shift, ib[2] + ih[2])
