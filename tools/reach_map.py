"""Print an ASCII reachability map for the downward grasp (numpy only; no Isaac app needed).

Run: docker/isaac/run.sh /isaac-sim/python.sh tools/reach_map.py [z ...]
"""
import sys

import numpy as np
import yaml

sys.path.insert(0, "src")
from agentic_sim.kinematics import So101Chain  # noqa: E402

cfg = yaml.safe_load(open("configs/teleop.yaml")) if len(sys.argv) > 99 else None
c = So101Chain.from_dict(yaml.safe_load(open("configs/so101_chain.yaml")))
c.ee_offset = np.array([0.0069, 0.0, -0.085])
pan = (0.0231, 0.0208)
zs = [float(v) for v in sys.argv[1:]] or [0.025, 0.06, 0.12]
ys = np.arange(-0.30, 0.31, 0.03)
for z in zs:
    print(f"z={z}  columns y from {ys[0]:.2f} to {ys[-1]:.2f} step 0.03 (left=+y is to the right of the printout)")
    for x in np.arange(0.04, 0.42, 0.02):
        row = ""
        for y in ys:
            yaw = np.arctan2(y - pan[1], x - pan[0]) + np.pi / 2
            ok = False
            a = np.arctan2(y - pan[1], x - pan[0]) - np.pi / 2  # pan angle guess (base reaches along -Y)
            for s in ([a, 0.0, 0.0, 1.5, 0.0], [a, -1.0, 1.0, 1.0, 0.0], [a, 0.5, -0.5, 1.5, 0.0], [a, 1.0, -1.2, 1.0, 0.0]):
                q, ok, info = c.solve_ik([x, y, z], yaw, np.array(s), iters=150)
                if ok:
                    break
            row += "#" if ok else "."
        print(f"x={x:.2f} {row}")
