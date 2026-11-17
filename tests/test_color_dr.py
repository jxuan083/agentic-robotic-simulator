"""Colour DR sampler (numpy only): block / arm draws keep their sequence when blocks are added."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from agentic_sim.color_dr import ColorSampler, Palette  # noqa: E402


def test_extra_blocks_do_not_shift_first_block_and_arm_within_a_draw():
    pal = Palette.load("configs/palettes/block_colors.yaml")
    one = ColorSampler(pal, 7).next(1)
    eight = ColorSampler(pal, 7).next(8)
    assert one["block"] == eight["block"] and one["arm_shell"] == eight["arm_shell"]
    assert "extra_blocks" not in one and len(eight["extra_blocks"]) == 7


def test_same_seed_same_sequence():
    pal = Palette.load("configs/palettes/block_colors.yaml")
    a, b = ColorSampler(pal, 3), ColorSampler(pal, 3)
    assert [a.next(4) for _ in range(5)] == [b.next(4) for _ in range(5)]
