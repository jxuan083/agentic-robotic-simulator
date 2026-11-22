"""Seeded, bounded block spawning shared by both engines (mount frame, metres).

Sampling uses bounding circles by default or rotated footprints when configured. It
checks support boundaries, tray obstacles, base clearance, pair separation and,
when enabled by the session adapter, grasp/approach IK. It is not path planning.
"""
from __future__ import annotations

import copy
import math
import random
import secrets
from collections import Counter
from dataclasses import replace

from agentic_sim.config import load, v
from agentic_sim.scene_layout import TRAYS, palette_by_name, table_box, tray_parts

REGION_TRAY = {"storage": "storage_pink", "placement": "placement_beige"}

DEFAULT_CONFIG = "configs/random_blocks.yaml"


def _local(xy, center, yaw):
    dx, dy = xy[0] - center[0], xy[1] - center[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _footprint_half_extent(edge, yaw):
    return edge / 2 * (abs(math.cos(yaw)) + abs(math.sin(yaw)))


def _separated_cubes(xy, yaw, other, other_yaw, edge, gap):
    # A separating axis with this much interval clearance gives at least that
    # much Euclidean clearance for the two rotated footprints.
    dx, dy = xy[0] - other[0], xy[1] - other[1]
    for angle in (yaw, yaw + math.pi / 2, other_yaw, other_yaw + math.pi / 2):
        projected = abs(dx * math.cos(angle) + dy * math.sin(angle))
        widths = _footprint_half_extent(edge, angle - yaw) + _footprint_half_extent(edge, angle - other_yaw)
        if projected >= widths + gap:
            return True
    return False


def legal_position(scene, xy, placed, region, settings, *, yaw_deg=None, placed_yaws=None):
    """Conservative support / obstacle checks; every cube corner fits the support."""
    edge = float(v(scene["block"]["edge"]))
    radius = edge / math.sqrt(2)
    margin = float(settings["boundary_margin_m"])
    gap = float(settings["gap_m"])
    tb = table_box(scene)
    if any(abs(xy[k] - tb["pos"][k]) + radius + margin > tb["size"][k] / 2 for k in (0, 1)):
        return False
    bx, by, _ = v(scene["robot"]["base_pos"])
    distance = math.hypot(xy[0] - bx, xy[1] - by)
    if distance < float(settings["base_clearance_m"]) + radius:
        return False
    if distance + radius > float(settings["max_radius_m"]):
        return False
    bounds = settings["workspace_xy"]
    if not all(bounds[k][0] + radius <= xy[k] <= bounds[k][1] - radius for k in (0, 1)):
        return False
    yaw = None if yaw_deg is None else math.radians(yaw_deg)
    if yaw is None:
        if any(math.dist(xy, other) < 2 * radius + gap for other in placed):
            return False
    elif any(not _separated_cubes(xy, yaw, other, math.radians(angle), edge, gap)
             for other, angle in zip(placed, placed_yaws, strict=True)):
        return False
    if region in REGION_TRAY:
        info = tray_parts(scene["trays"][REGION_TRAY[region]])[1]
        local = _local(xy, info["center"], info["yaw"])
        half = radius if yaw is None else _footprint_half_extent(edge, yaw - info["yaw"])
        if any(abs(local[k]) + half + margin > info["inner_half"][k] for k in (0, 1)):
            return False
        # If trays overlap, do not assume their floors/walls form a valid support.
        obstacles = (name for name in TRAYS if name != REGION_TRAY[region])
    elif region == "table":
        obstacles = TRAYS
    else:
        raise ValueError("random region must be storage, placement or table")
    for name in obstacles:
        spec = scene["trays"][name]
        local = _local(xy, v(spec["center_xy"]), math.radians(float(v(spec["yaw_deg"]))))
        size = v(spec["outer_size"])
        if all(abs(local[k]) < size[k] / 2 + radius + margin for k in (0, 1)):
            return False
    return True


def sample_scene(scene, settings, *, seed, episode=0, region="storage", reachable=None):
    """Return a copied SceneSpec, colour names and an auditable spawn record.

    Never returns a partial layout: infeasible requests raise after bounded attempts.
    """
    if region not in REGION_TRAY and region != "table":
        raise ValueError("random region must be storage, placement or table")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("random seed must be a nonnegative integer")
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
        raise ValueError("random episode must be a nonnegative integer")
    lo, hi = settings["count_range"]
    if any(isinstance(n, bool) or not isinstance(n, int) for n in (lo, hi)) or not 1 <= lo <= hi <= int(scene["block"].get("max_count", 1)):
        raise ValueError("count_range must fit the scene's 1..max_count")
    yaw_range = settings.get("yaw_range_deg", [-180.0, 180.0])
    if (not isinstance(yaw_range, (list, tuple)) or len(yaw_range) != 2
            or not all(math.isfinite(float(a)) for a in yaw_range)
            or float(yaw_range[0]) > float(yaw_range[1])):
        raise ValueError("yaw_range_deg must be an increasing finite pair")
    for key in ("boundary_margin_m", "gap_m", "base_clearance_m", "max_radius_m"):
        if not math.isfinite(float(settings[key])) or float(settings[key]) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    bounds = settings["workspace_xy"]
    if len(bounds) != 2 or any(len(b) != 2 or not all(math.isfinite(float(x)) for x in b) or b[0] >= b[1] for b in bounds):
        raise ValueError("workspace_xy must contain increasing finite x/y bounds")
    attempts = settings["max_attempts_per_layout"]
    restarts = settings["layout_retries"]
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (attempts, restarts)):
        raise ValueError("sampling limits must be positive integers")
    names = list(palette_by_name(scene["block"]["extra_palette"]))
    allowed = settings.get("colors") or names
    if not isinstance(allowed, list) or any(c not in names for c in allowed):
        raise ValueError("colors must be a list of palette names")
    rng = random.Random(f"{seed}:{episode}")
    fixed = settings.get("fixed_colors")
    if fixed is None:
        count = rng.randint(lo, hi)
        colors = [rng.choice(allowed) for _ in range(count)]
    else:
        if not isinstance(fixed, list) or not lo <= len(fixed) <= hi or any(c not in names for c in fixed):
            raise ValueError("fixed_colors must name one palette color per block and fit count_range")
        colors = list(fixed)
        count = len(colors)
    rotated = settings.get("yaw_aware_footprints", False)
    edge = float(v(scene["block"]["edge"]))
    floor = tray_parts(scene["trays"][REGION_TRAY[region]])[1]["floor_top"] if region in REGION_TRAY else 0.0
    for _ in range(restarts):
        positions = []
        yaws = []
        for _ in range(attempts):
            if region in REGION_TRAY:
                info = tray_parts(scene["trays"][REGION_TRAY[region]])[1]
                lx, ly = (rng.uniform(-half, half) for half in info["inner_half"])
                c, s = math.cos(info["yaw"]), math.sin(info["yaw"])
                xy = (info["center"][0] + c * lx - s * ly, info["center"][1] + s * lx + c * ly)
            else:
                xy = tuple(rng.uniform(*b) for b in bounds)
            yaw = rng.uniform(float(yaw_range[0]), float(yaw_range[1])) if rotated else None
            if not legal_position(scene, xy, positions, region, settings, yaw_deg=yaw, placed_yaws=yaws):
                continue
            z = floor + edge / 2 + 0.0005
            if yaw is None:
                yaw = rng.uniform(float(yaw_range[0]), float(yaw_range[1]))
            if reachable is not None:
                try:
                    can_reach = reachable((*xy, z), yaw)
                except TypeError:  # compatibility with older one-argument test/client callbacks
                    can_reach = reachable((*xy, z))
                if not can_reach:
                    continue
            positions.append(xy)
            yaws.append(yaw)
            if len(positions) == count:
                out = copy.deepcopy(scene)
                out["block"].update(count=count, init_xy=list(positions[0]), layout_xy=[list(p) for p in positions],
                                    init_yaw_deg=yaws[0], init_yaw_degs=yaws, _spawn_floor_top=floor)
                record = {"seed": seed, "episode": episode, "region": region, "count": count,
                          "colors": colors, "counts": dict(Counter(colors)),
                          "positions_m": [[*p, z] for p in positions], "yaw_deg": yaws[0], "yaw_degs": yaws,
                          "ik_checked": reachable is not None,
                          "ik_orientation": ("cube_edge_aligned_mod_90" if getattr(reachable, "checks_yaw", False)
                                             else "position_only" if reachable is not None else "not_checked")}
                return out, colors, record
    raise ValueError(f"Cannot place {count} blocks legally in {region}; enlarge the region, reduce count/gap, "
                     "or check IK settings. No partial scene was created.")


def grasp_ik_checker(scene, teleop_path):
    """reachable(block_centre_xyz, yaw_deg) -> bool: the scripted grasp's approach (6 cm above, at least the 9 cm
    transport height) and grasp point both have a tilted-grasp IK solution with the fingers on a pair of faces
    (any of the cube's four quarter turns). Kinematics only: no self-collision or table contact."""
    import numpy as np
    from agentic_sim.kinematics import So101Chain
    from agentic_sim.teleop.ee_target import load_controller
    ctrl = load_controller(So101Chain.from_dict(load("configs/so101_chain.yaml")), load(teleop_path),
                           base_pos=v(scene["robot"]["base_pos"]), scene=scene)
    def reachable(pos, yaw_deg=0.0):
        for symmetry in range(4):
            q = ctrl.q_home.copy()
            ok = True
            for z in (max(0.09, pos[2] + 0.06), max(float(ctrl.p.ws_lo[2]), pos[2] + 0.008)):
                target = np.array([pos[0], pos[1], z])
                if np.any(target < ctrl.p.ws_lo) or np.any(target > ctrl.p.ws_hi):
                    ok = False
                    break
                absolute_yaw = math.radians(yaw_deg) + symmetry * math.pi / 2
                radial_baseline = math.atan2(target[1] - ctrl.pan[1], target[0] - ctrl.pan[0]) + ctrl.p.finger_heading
                yaw_offset = absolute_yaw - radial_baseline
                q, solved, _ = ctrl.chain.solve_ik_tilt(target, ctrl.yaw_for(target, yaw_offset), q,
                                                         ctrl.p.max_tilt_deg)
                if not solved:
                    ok = False
                    break
            if ok:
                return True
        return False
    reachable.checks_yaw = True
    return reachable


def prepare_session(cfg, scene, log=print):
    """Resolve a seed once, then materialize the layout before engine creation."""
    if not cfg.random_blocks:
        return cfg, scene, None, None
    if cfg.block_colors or cfg.task or cfg.color_dr or load(cfg.color_dr_config).get("enabled"):
        raise ValueError("random blocks cannot combine with explicit block_colors, task scenarios or color DR")
    seed = secrets.randbits(63) if cfg.random_seed is None else cfg.random_seed
    settings = load(cfg.random_config)
    checker = grasp_ik_checker(scene, cfg.teleop) if settings.get("check_ik", True) else None
    scene, colors, record = sample_scene(scene, settings, seed=seed, episode=cfg.random_episode,
                                          region=cfg.random_region, reachable=checker)
    cfg = replace(cfg, random_seed=seed, blocks=record["count"])
    log(f"random blocks: {record}")
    return cfg, scene, colors, record


def add_arguments(parser):
    parser.add_argument("--random-blocks", "--random_blocks", action="store_true", help="resample block counts, colours and legal positions at startup and Reset")
    parser.add_argument("--random-seed", "--random_seed", type=int, default=None, help="repeatable spawn sequence (default: a fresh seed)")
    parser.add_argument("--random-region", "--random_region", choices=("storage", "table"), default="storage")
    parser.add_argument("--random-config", "--random_config", default=DEFAULT_CONFIG)
