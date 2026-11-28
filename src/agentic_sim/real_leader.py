"""Driven leader hand-over for the collection desk (same order as the route tool and LeHome's DAgger recorder).

The follower holds the standard start. The leader is driven to the same pose (along the taught park -> start
route when it rests folded, gripper last, half torque), the operator takes it during a countdown, its torque is
released while the follower keeps holding, and leader motion during a short guard (gravity sag) is ignored: the
collection's relative leader mapping starts from the released pose.
"""

from agentic_sim.real_control import JOINTS, pose

BODY = JOINTS[:-1]
SPEED = dict.fromkeys(BODY, 20.0) | {"gripper": 40.0}
ARRIVED = dict.fromkeys(BODY, 3.0) | {"gripper": 8.0}
STALL_LAG, STALL_S = 8.0, 0.5
SETTLE_S = 3.0
ROUTE_NEAR = 25.0
COUNTDOWN_S = 3
GUARD_S = 0.8
DROPPED = 12.0


def gap(a, b):
    return max(abs(a[j] - b[j]) for j in BODY)


def goals(start, reference, route=None):
    hold = {"gripper": start["gripper"]}
    out = []
    if route and gap(start, route[0]) <= ROUTE_NEAR and gap(route[-1], reference) <= ROUTE_NEAR:
        out = [pose(p) | hold for p in route[1:]]
    return [*out, pose(reference) | hold, pose(reference)]


def handover(leader, reference, *, route, clock, wait, hz, say, halted, event):
    """Returns the leader pose after release. Raises RuntimeError when blocked, dropped or stopped."""
    current = pose(leader.read())
    plan = goals(current, reference, route)
    event("leader_drive_plan", waypoints=len(plan), via_route=len(plan) > 2)
    leader.drive_on()
    event("leader_torque_on", leader=current)
    try:
        goal, stall_since, reached_at, next_say = dict(current), None, None, 0
        say("leader 自動移到 follower 的姿勢，手先離開 leader")
        while True:
            began = clock()
            if halted():
                leader.drive(pose(leader.read()))
                raise RuntimeError("operator stopped during the leader hand-over; support the leader")
            aim = plan[0]
            for joint in JOINTS:
                step = SPEED[joint] / hz
                goal[joint] += max(-step, min(step, aim[joint] - goal[joint]))
            if len(plan) > 1 and all(goal[j] == aim[j] for j in JOINTS):
                plan.pop(0)
            leader.drive(goal)
            now = pose(leader.read())
            if max(gap(goal, now), abs(goal["gripper"] - now["gripper"]) / 3) > STALL_LAG:
                stall_since = stall_since or began
                if began - stall_since > STALL_S * 1e9:
                    leader.drive(now)
                    event("leader_stall", goal=goal, leader=now)
                    raise RuntimeError("leader blocked on its way to the follower pose; support it and retry")
            else:
                stall_since = None
            if all(goal[j] == reference[j] for j in JOINTS):
                reached_at = reached_at or began
                if all(abs(now[j] - reference[j]) <= ARRIVED[j] for j in JOINTS):
                    event("leader_arrived", leader=now)
                    break
                if began - reached_at > SETTLE_S * 1e9:
                    raise RuntimeError("leader could not reach the follower pose; support it and retry")
            wait(max(0.0, began / 1e9 + 1 / hz - clock() / 1e9))
        end = clock() + COUNTDOWN_S * 1e9
        while (began := clock()) < end:
            if halted():
                raise RuntimeError("operator stopped during the leader countdown; support the leader")
            if began >= next_say:
                say(f"握住 leader：{int((end - began) / 1e9) + 1} 秒後解 leader 扭力，由你接手")
                next_say = began + 1_000_000_000
            wait(max(0.0, began / 1e9 + 1 / hz - clock() / 1e9))
    finally:
        leader.release()
        event("leader_torque_off", leader=pose(leader.read()))
    end = clock() + GUARD_S * 1e9
    while (began := clock()) < end:
        wait(max(0.0, began / 1e9 + 1 / hz - clock() / 1e9))
    after = pose(leader.read())
    sag = gap(after, reference)
    event("leader_engaged", leader=after, sag_deg=sag)
    if sag > DROPPED:
        raise RuntimeError(f"leader moved {sag:.0f} deg after release (not supported); retry the recording")
    say("已交接，可以操作 leader")
    return after
