"""Persistent single-arm task cards, readback decisions and replay history."""

from collections import Counter
from contextlib import contextmanager
import fcntl
import json
from math import dist, isfinite
from pathlib import Path
from random import Random

from agentic_sim.real_episode import Episode, write_json
from agentic_sim.real_review import confirmed_frames, review_episode
from agentic_sim.real_round import capture_round, replay_round, validate_setup


COLORS = ("red", "blue", "yellow")
TRAYS = ("pink", "beige")
REGIONS = ("near", "far")
GROUP_CELL_MM = 60


def position(values):
    if len(values) != 3 or any(type(v) not in (int, float) or not isfinite(v) for v in values):
        raise ValueError("tray position needs x mm, y mm and yaw degrees")
    return list(values)


def make_cards(layout):
    random = Random(layout["seed"])
    separation = layout["reserve_distance_mm"]
    if type(separation) not in (int, float) or not isfinite(separation) or separation <= 0:
        raise ValueError("reserve_distance_mm must be positive")
    candidates = {}
    for tray in TRAYS:
        for region in REGIONS:
            zone = f"{tray}_{region}"
            points = {tuple(position(point)) for point in layout["zones"][zone]}
            candidates[tray, region] = sorted(points)
    reserves = []
    for (tray, region), points in candidates.items():
        random.shuffle(points)
        eligible = [p for p in points if all(dist(p[:2], q["pose"][:2]) >= separation
                                             for q in reserves if q["tray"] == tray)]
        for point in eligible:
            if sum(row["tray"] == tray and row["region"] == region for row in reserves) == 5:
                break
            if all(dist(point[:2], row["pose"][:2]) >= separation for row in reserves if row["tray"] == tray):
                reserves.append({"id": f"E{len(reserves) + 1:02}", "tray": tray,
                                 "region": region, "pose": point})
        if sum(row["tray"] == tray and row["region"] == region for row in reserves) != 5:
            raise ValueError(f"{tray} {region}: five separated held-out positions are needed")
    for (tray, region), points in candidates.items():
        usable = [p for p in points if all(dist(p[:2], row["pose"][:2]) >= separation
                                           for row in reserves if row["tray"] == tray)]
        distinct = {}
        for point in usable:
            distinct.setdefault(point[:2], []).append(point)
        if not distinct:
            raise ValueError(f"{tray} {region}: no approved training position outside the held-out margin")
        candidates[tray, region] = list(distinct.values())

    # Every card gets its own block position: the grasp does not depend on the colour, so spreading the
    # 120 demonstrations over the tray covers more arm configurations than repeating a position per colour.
    # Positions are drawn without replacement per tray and region (reused only if a zone has too few), and
    # nearby positions share a layout group (GROUP_CELL_MM cells) so train / development splits stay apart.
    pools = {}
    for key, groups in candidates.items():
        order = list(groups)
        random.shuffle(order)
        pools[key] = order
    used = Counter()
    cards = []
    for i in range(20):
        region = REGIONS[i // 10]
        kind = "normal" if i % 10 < 6 else "small_offset"
        for tray, destination in (TRAYS, TRAYS[::-1]):
            for color in COLORS:
                pool = pools[tray, region]
                target = random.choice(pool[used[tray, region] % len(pool)])
                used[tray, region] += 1
                cell = f"{int(target[0] // GROUP_CELL_MM)}-{int(target[1] // GROUP_CELL_MM)}"
                cards.append({"id": f"D{len(cards) + 1:03}", "source_tray": tray, "destination": destination,
                              "color": color, "region": region, "kind": kind, "pose": list(target),
                              "layout_group": f"{tray}-{region}-{cell}"})
    return cards, reserves


def _held_out_map(layout, reserves):
    panels = []
    for tray in TRAYS:
        points = [p for region in REGIONS for p in layout["zones"][f"{tray}_{region}"]]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        panel = []
        for row in reserves:
            if row["tray"] != tray:
                continue
            x, y, yaw = row["pose"]
            left = 5 + 90 * (x - min(xs)) / max(1, max(xs) - min(xs))
            bottom = 8 + 84 * (y - min(ys)) / max(1, max(ys) - min(ys))
            panel.append(f'<span style="left:{left:.2f}%;bottom:{bottom:.2f}%" '
                         f'title="{x:g} mm, {y:g} mm, yaw {yaw:g}°">{row["id"]}</span>')
        panels.append(f'<section><h2>{tray}</h2><div class="tray">{"".join(panel)}</div></section>')
    return ('<!doctype html><meta charset="utf-8"><title>保留擺位</title>'
            '<style>body{font:16px system-ui;margin:24px;color:#20272d}main{display:flex;gap:30px;flex-wrap:wrap}'
            '.tray{position:relative;width:330px;height:230px;background:#eee8de;border:2px solid #a89a87;'
            'border-radius:15px}.tray span{position:absolute;transform:translate(-50%,50%);background:#fff;'
            'border:1px solid #534537;border-radius:12px;padding:3px 5px;font-size:12px}</style>'
            '<h1>保留擺位 E01–E20</h1><p>盤相對座標，右方 +x，上方 +y。游標移到點上查看座標和 yaw。</p>'
            f'<main>{"".join(panels)}</main>')


@contextmanager
def locked_session(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this session is already open") from error
        yield


class Collection:
    def __init__(self, root, backend, plan, layout):
        validate_setup(backend, plan)
        self.root = Path(root)
        self.path = self.root / "session.json"
        self.backend = backend
        self.plan = plan
        self.layout = layout
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if (self.state["source"], self.state["setup"], self.state["layout"]) != (
                backend.source, plan.reference(), layout
            ):
                raise ValueError("session belongs to a different source, motion or card layout setup")
            self._recover()
        else:
            cards, reserves = make_cards(layout)
            self.state = {"source": backend.source, "setup": plan.reference(), "cards": cards, "held_out": reserves,
                          "attempts": [], "replays": [], "dagger": [], "first_replay_episode": None,
                          "operator_recovery_required": False, "layout": layout, "preflight": []}
            self.save()
            write_json(self.root / "held_out.json", reserves)
            (self.root / "held_out.html").write_text(_held_out_map(layout, reserves), encoding="utf-8")

    def save(self):
        write_json(self.path, self.state)
        by_group = {}
        for entry in self.state["attempts"]:
            if entry["status"] == "accepted":
                group = entry["layout_group"]
                by_group.setdefault(group, []).append(entry["episode"])
        groups = sorted({row["layout_group"] for row in self.state["cards"]})
        Random(self.layout["seed"]).shuffle(groups)
        dev_groups = set(groups[:max(1, len(groups) // 5)])
        names = ("train", "development")
        splits = {name: [] for name in names}
        for group, entries in by_group.items():
            splits["development" if group in dev_groups else "train"].extend(entries)
        write_json(self.root / "dataset_groups.json", {"groups": by_group, **splits})

    def _recover(self):
        for bucket in ("attempts", "replays", "dagger", "preflight"):
            for entry in self.state[bucket]:
                if entry["status"] != "running":
                    continue
                entry["status"] = "interrupted"
                entry["error"] = "session closed before the operation was finalized"
                self.state["operator_recovery_required"] = True
        self.save()

    def clear_recovery(self):
        self.state["operator_recovery_required"] = False
        self.save()

    def _check_ready(self):
        if self.state["operator_recovery_required"]:
            raise RuntimeError("verify follower support and torque state on site before resuming")

    def card(self, card_id):
        return next(card for card in self.state["cards"] if card["id"] == card_id)

    def progress(self):
        accepted = [entry for entry in self.state["attempts"] if entry["status"] == "accepted"]
        counts = Counter((self.card(entry["card"])["source_tray"], self.card(entry["card"])["color"])
                         for entry in accepted)
        return len(accepted), counts

    def current(self):
        done = {entry["card"] for entry in self.state["attempts"] if entry["status"] == "accepted"}
        card = next((card for card in self.state["cards"] if card["id"] not in done), None)
        pending = next((entry for entry in reversed(self.state["attempts"])
                        if card and entry["card"] == card["id"]), None)
        return card, pending

    def _confirm_layout(self, card, observed):
        actual = position(observed) if observed is not None else None
        group = card["layout_group"]
        if actual is not None:
            for row in self.state["held_out"]:
                if row["tray"] == card["source_tray"] and dist(row["pose"][:2], actual[:2]) < self.layout["reserve_distance_mm"]:
                    raise ValueError(f"placement overlaps held-out layout {row['id']}; reposition the block")
            nearest = min((row for row in self.state["cards"] if row["source_tray"] == card["source_tray"]),
                          key=lambda row: dist(row["pose"][:2], actual[:2]))
            group = nearest["layout_group"]
        return dict(card, actual_pose=actual, layout_confirmed=actual is not None, layout_group=group)

    def record(self, operator, observed=None):
        self._check_ready()
        if self.backend.source == "real" and not any(entry["status"] == "complete" for entry in self.state["preflight"]):
            raise ValueError("complete an empty-table motion check for this setup first")
        card, pending = self.current()
        if card is None:
            raise ValueError("all demonstration cards are complete")
        if pending and pending["status"] in ("needs_review", "needs_replay"):
            raise ValueError("confirm or retry the pending episode first")
        recorded_card = self._confirm_layout(card, observed)
        attempt = 1 + sum(entry["card"] == card["id"] for entry in self.state["attempts"])
        entry = {"card": card["id"], "episode": f"episodes/{card['id']}-{attempt:02}",
                 "status": "running", "reviewed": False}
        self.state["attempts"].append(entry)
        self.save()
        try:
            captured = capture_round(self.root / entry["episode"], self.backend, self.plan, operator,
                                     card=recorded_card)
        except BaseException as error:
            self._interrupted(entry, error)
            raise
        try:
            report = review_episode(self.root / entry["episode"], source=self.backend.source, arm=self.plan.arm,
                                    card=recorded_card, rgb_depth=self.plan.rgb_depth)
        except (OSError, KeyError, TypeError, ValueError) as error:
            entry.update(status="retry", error=f"readback: {error}")
        else:
            entry.update(report=f"{entry['episode']}/readback.json", opening_global=report["opening_global"],
                         review_reasons=report["review_reasons"], readback_ok=report["ok"])
            if not report["ok"] or captured.get("cancelled"):
                entry["status"] = "retry"
            elif not self.state["first_replay_episode"] or report["review_reasons"]:
                entry["status"] = "needs_review"
            else:
                self.approve(entry["episode"])
        self.save()
        return entry

    def _interrupted(self, entry, error):
        entry.update(status="interrupted", error=str(error))
        self.state["operator_recovery_required"] = True
        self.save()

    def episode_entry(self, source):
        entry = next((entry for entry in self.state["attempts"] if entry["episode"] == source), None)
        if entry is None or not entry.get("readback_ok"):
            raise ValueError("the episode needs a valid disk readback")
        return entry

    def approve(self, source, selected=None, *, visual=None):
        entry = self.episode_entry(source)
        if entry["status"] in ("retry", "interrupted"):
            raise ValueError("this attempt is no longer eligible")
        directory = self.root / source
        report = json.loads((directory / "readback.json").read_text(encoding="utf-8"))
        episode = Episode(directory)
        card = episode.manifest["task_card"]
        correction = directory / "layout_review.json"
        if correction.is_file():
            card = json.loads(correction.read_text(encoding="utf-8"))
        if not card.get("layout_confirmed"):
            raise ValueError("verify actual placement coordinates and held-out margin first")
        outcome = visual if visual is not None else report["operator_result"]
        if outcome.get("color") != card["color"] or outcome.get("placed") is not True:
            raise ValueError("effective demonstration requires confirmed correct color and placement")
        if selected is None:
            selected = {key: value["frame"] for key, value in report["candidates"].items()}
        frames = confirmed_frames(episode, selected)
        write_json(directory / "review.json", {"frames": frames, "outcome": outcome,
                                               "training_range": [frames["handoff"]["frame"], frames["lift"]["frame"]]})
        entry["reviewed"] = True
        entry["layout_group"] = card["layout_group"]
        entry["status"] = "accepted" if self.state["first_replay_episode"] else "needs_replay"
        self._complete_first(entry)
        self.save()

    def verify_layout(self, source, observed):
        entry = self.episode_entry(source)
        layout = self._confirm_layout(self.card(entry["card"]), observed)
        if not layout["layout_confirmed"]:
            raise ValueError("actual placement needs x, y and yaw")
        write_json(self.root / source / "layout_review.json", layout)

    def retry(self, source):
        entry = self.episode_entry(source)
        if entry["status"] not in ("needs_review", "needs_replay"):
            raise ValueError("only a pending episode can be retried")
        entry["status"] = "retry"
        self.save()

    def replay(self, source, operator, *, layout_ready):
        self._check_ready()
        entry = self.episode_entry(source)
        if entry["status"] not in ("accepted", "needs_review", "needs_replay"):
            raise ValueError("select a valid active or accepted episode")
        if layout_ready is not True:
            raise ValueError("restore the scene and explicitly confirm clearance first")
        record = {"source_episode": source, "record": f"replays/R{len(self.state['replays']) + 1:04}",
                  "status": "running"}
        self.state["replays"].append(record)
        self.save()
        try:
            episode = Episode(self.root / source)
            result = replay_round(episode, self.backend, self.plan, self.root / record["record"], operator)
        except BaseException as error:
            self._interrupted(record, error)
            raise
        record.update(result)
        outcome = result.get("outcome", {})
        first = not self.state["first_replay_episode"] and entry["card"] == self.state["cards"][0]["id"]
        record["passed"] = (result["status"] == "complete" and outcome.get("placed") is True
                            and outcome.get("color") == self.card(entry["card"])["color"])
        if first and record["passed"]:
            entry["replay_passed"] = True
        self._complete_first(entry)
        self.save()
        return record

    def _complete_first(self, entry):
        if entry["reviewed"] and entry.get("replay_passed") and not self.state["first_replay_episode"]:
            self.state["first_replay_episode"] = entry["episode"]
            entry["status"] = "accepted"

    def dagger(self, operator, policy):
        self._check_ready()
        entry = {"episode": f"dagger/P{len(self.state['dagger']) + 1:04}", "status": "running"}
        self.state["dagger"].append(entry)
        self.save()
        try:
            outcome = capture_round(self.root / entry["episode"], self.backend, self.plan, operator, policy=policy)
        except BaseException as error:
            self._interrupted(entry, error)
            raise
        report = review_episode(self.root / entry["episode"], source=self.backend.source, arm=self.plan.arm)
        entry.update(status="complete" if report["ok"] and not outcome.get("cancelled") else "retry",
                     takeovers=report["takeovers"])
        self.save()
        return entry

    def preflight(self, operator):
        self._check_ready()
        entry = {"episode": f"preflight/T{len(self.state['preflight']) + 1:04}", "status": "running"}
        self.state["preflight"].append(entry)
        self.save()
        try:
            outcome = capture_round(self.root / entry["episode"], self.backend, self.plan, operator)
        except BaseException as error:
            self._interrupted(entry, error)
            raise
        report = Episode(self.root / entry["episode"]).validate()
        entry["status"] = "complete" if report["ok"] and not outcome.get("cancelled") else "retry"
        entry["issues"] = report["issues"]
        self.save()
        return entry
