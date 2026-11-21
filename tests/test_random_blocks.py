"""Geometric invariants and replay/reset contracts for shared random spawning."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from agentic_sim.config import load, v
from agentic_sim.random_blocks import legal_position, prepare_session, sample_scene, grasp_ik_checker
from agentic_sim.scene_layout import block_layout, tray_parts, table_box
from agentic_sim.launch import RunOptions, _app_args
from agentic_sim.experiments import load_experiment


@pytest.mark.parametrize('scene_path', ['configs/scenes/lab_current_minimal.yaml', 'configs/scenes/lab_20261120.yaml'])
@pytest.mark.parametrize('region', ['storage', 'table'])
def test_many_seeds_have_supported_nonoverlapping_blocks(scene_path, region):
    scene = load(scene_path)
    original = copy.deepcopy(scene)
    settings = load('configs/random_blocks.yaml')
    seen_counts, seen_colors = set(), set()
    for seed in range(60):
        out, colors, record = sample_scene(scene, settings, seed=seed, region=region)
        lay = block_layout(out, colors=colors)
        seen_counts.add(len(colors)); seen_colors.update(colors)
        assert len(lay.init_pos) == record['count'] == len(colors)
        assert lay.init_pos == [tuple(p) for p in record['positions_m']]
        assert sum(record['counts'].values()) == len(colors)
        placed = []
        for x, y, z in lay.init_pos:
            assert legal_position(scene, (x,y), placed, region, settings)
            placed.append((x,y))
            floor = tray_parts(scene['trays']['storage_pink'])[1]['floor_top'] if region == 'storage' else 0
            assert z - lay.edge / 2 == pytest.approx(floor + .0005)
            # Independent corner check in table coordinates (all four corners).
            tb = table_box(scene)
            for dx in (-lay.edge/2, lay.edge/2):
                for dy in (-lay.edge/2, lay.edge/2):
                    assert abs(x+dx-tb['pos'][0]) < tb['size'][0]/2
                    assert abs(y+dy-tb['pos'][1]) < tb['size'][1]/2
            for ox,oy in placed[:-1]:
                assert math.hypot(x-ox,y-oy) >= math.sqrt(2)*lay.edge+settings['gap_m']
        assert scene == original
    assert seen_counts == set(range(1,9))
    assert len(seen_colors) == 8


def test_seed_replay_and_episode_and_infeasible_requests():
    scene, settings = load('configs/scenes/lab_20261120.yaml'), load('configs/random_blocks.yaml')
    a = sample_scene(scene, settings, seed=101)
    assert a == sample_scene(scene, settings, seed=101)
    assert a != sample_scene(scene, settings, seed=101, episode=1)
    assert len(a[2]['yaw_degs']) == a[2]['count']
    assert a[2]['yaw_degs'] == sample_scene(scene, settings, seed=101)[2]['yaw_degs']
    assert len(set(a[2]['yaw_degs'])) > 1
    assert all(-180 <= yaw <= 180 for yaw in a[2]['yaw_degs'])
    rotations = block_layout(a[0], colors=a[1]).init_rots
    assert len(rotations) == a[2]['count']
    assert rotations[0] != rotations[1]
    bad = dict(settings, count_range=[8,8], workspace_xy=[[.100,.101],[.100,.101]], max_attempts_per_layout=5,layout_retries=2)
    with pytest.raises(ValueError, match='Cannot place'):
        sample_scene(scene,bad,seed=1)
    with pytest.raises(ValueError, match='No partial'):
        sample_scene(scene,dict(settings,max_attempts_per_layout=2,layout_retries=1),seed=1,reachable=lambda p:False)
    for seed in (-1, True, 1.1):
        with pytest.raises(ValueError, match='seed'):
            sample_scene(scene,settings,seed=seed)
    for invalid in ([], [1], [2, 1], [float('nan'), 1]):
        with pytest.raises(ValueError, match='yaw_range_deg'):
            sample_scene(scene, dict(settings, yaw_range_deg=invalid), seed=2)


def test_rotated_storage_tray_and_boundary_obstacle_checks():
    scene, settings = load('configs/scenes/lab_20261120.yaml'),load('configs/random_blocks.yaml')
    scene['trays']['storage_pink']['yaw_deg'] = 30
    settings['count_range'] = [8,8]
    out,colors,_ = sample_scene(scene, settings, seed=9)
    info=tray_parts(scene['trays']['storage_pink'])[1]
    for x,y,z in block_layout(out,colors=colors).init_pos:
        dx,dy=x-info['center'][0],y-info['center'][1]
        lx=math.cos(info['yaw'])*dx+math.sin(info['yaw'])*dy
        ly=-math.sin(info['yaw'])*dx+math.cos(info['yaw'])*dy
        assert abs(lx)+.0295/math.sqrt(2)+settings['boundary_margin_m'] <= info['inner_half'][0]
        assert abs(ly)+.0295/math.sqrt(2)+settings['boundary_margin_m'] <= info['inner_half'][1]
    assert not legal_position(scene,(0.,0.),[],'table',settings)
    assert not legal_position(scene,tuple(v(scene['trays']['placement_beige']['center_xy'])),[],'table',settings)


def test_sampled_targets_have_approach_and_grasp_ik():
    scene,settings=load('configs/scenes/lab_20261120.yaml'),load('configs/random_blocks.yaml')
    settings['count_range']=[8,8]
    check=grasp_ik_checker(scene,'configs/teleop.yaml')
    for region in ('storage','table'):
        _,_,rec=sample_scene(scene,settings,seed=8,region=region,reachable=check)
        assert rec['ik_checked']
        assert all(check(p) for p in rec['positions_m'])


def test_launch_propagates_random_settings():
    args=_app_args(load_experiment('lab_20261120'),RunOptions(random_blocks=True,random_seed=71,random_region='table'))
    assert '--random-blocks' in args
    assert args[args.index('--random-seed')+1]=='71'
    assert args[args.index('--random-region')+1]=='table'


def test_session_preparation_replay_and_reset_rebuild_contract():
    # Use the simulator-independent config fields to check startup preparation.
    from dataclasses import dataclass, replace
    @dataclass
    class Config:
        random_blocks: bool = True
        random_seed: int | None = 63
        random_episode: int = 0
        random_region: str = 'storage'
        random_config: str = 'configs/random_blocks.yaml'
        teleop: str = 'configs/teleop.yaml'
        color_dr_config: str = 'configs/color_randomization.yaml'
        color_dr: bool = False
        block_colors: list | None = None
        task: str | None = None
        blocks: int | None = None
    scene = load('configs/scenes/lab_20261120.yaml')
    cfg=Config()
    resolved, out, colors, record = prepare_session(cfg, scene, lambda _:None)
    assert resolved.blocks == len(colors)
    assert prepare_session(cfg,scene,lambda _:None)[3] == record
    assert prepare_session(replace(resolved,random_episode=1),scene,lambda _:None)[3] != record
    assert prepare_session(replace(cfg,random_blocks=False),scene,lambda _:None) == (replace(cfg,random_blocks=False),scene,None,None)
    for bad in (replace(cfg,block_colors=['red']),replace(cfg,task='some.yaml'),replace(cfg,color_dr=True)):
        with pytest.raises(ValueError,match='cannot combine'):
            prepare_session(bad,scene)
    fresh=prepare_session(replace(cfg,random_seed=None),scene,lambda _:None)
    assert fresh[0].random_seed == fresh[3]['seed']
