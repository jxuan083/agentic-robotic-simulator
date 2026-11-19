"""Requirement ledger: what is wanted, what has been delivered (verified), and what is left.

  led = Ledger()
  led.set_order({"red": 3}, "total", t)      # revision 1
  led.verified("red", +1, t, block=0)        # only after an external check, never on a skill's own report
  led.set_order({"red": 1, "blue": 2}, "total", t)   # revision 2: a new total, not an addition
  led.need("blue") -> 2 ; led.over("red") -> 0

Every change is an event (dict) with the revision it belongs to; Ledger.replay(events) rebuilds the same ledger from
a log. Holding (a block in the gripper) is executor state and is added to snapshots by the executor.
"""

from __future__ import annotations

import copy

MODES = ("total", "add", "set")


class Ledger:
    def __init__(self):
        self.revision = 0
        self.target: dict = {}          # colour -> wanted count (current revision)
        self.delivered: dict = {}       # colour -> verified count in the placement tray
        self.accepted_extra: dict = {}  # colour -> over-delivery the user approved ("deliver anyway")
        self.uncertain: dict = {}       # colour -> outcomes that could not be verified
        self.events: list = []

    # ---- queries ------------------------------------------------------------------------
    def need(self, color: str) -> int:
        return max(0, self.target.get(color, 0) - self.delivered.get(color, 0))

    def over(self, color: str) -> int:
        """Delivered beyond the order (and not approved by the user): to be retrieved."""
        return max(0, self.delivered.get(color, 0) - self.target.get(color, 0) - self.accepted_extra.get(color, 0))

    def colors(self) -> list:
        """Every colour in the order or in the tray, order first (insertion order)."""
        out = list(self.target)
        out += [c for c in self.delivered if c not in out]
        return out

    def needed(self) -> dict:
        return {c: self.need(c) for c in self.colors() if self.need(c) > 0}

    def overs(self) -> dict:
        return {c: self.over(c) for c in self.colors() if self.over(c) > 0}

    def done(self) -> bool:
        return self.revision > 0 and not self.needed() and not self.overs()

    def snapshot(self) -> dict:
        nz = lambda d: {k: v for k, v in d.items() if v}  # noqa: E731
        return {"revision": self.revision, "target": dict(self.target), "delivered": nz(self.delivered),
                "need": self.needed(), "over": self.overs(), "accepted_extra": nz(self.accepted_extra),
                "uncertain": nz(self.uncertain), "done": self.done()}

    # ---- changes ------------------------------------------------------------------------
    def _event(self, t: float, kind: str, **data) -> dict:
        e = {"t": round(float(t), 4), "rev": self.revision, "kind": kind, **data, "ledger": self.snapshot()}
        self.events.append(e)
        return e

    @staticmethod
    def compose(target: dict, counts: dict, mode: str) -> dict:
        """The order after applying `counts` with `mode` to `target`:
        'total' the new complete order (colours left out are no longer wanted); 'add' added to the current order;
        'set' only the named colours change, to the given counts (0 removes the colour), the others are kept."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        counts = {str(c): int(n) for c, n in (counts or {}).items()}
        if any(n < 0 for n in counts.values()):
            raise ValueError(f"negative count in {counts}")
        if mode == "total":
            new = dict(counts)
        elif mode == "add":
            new = dict(target)
            for c, n in counts.items():
                new[c] = new.get(c, 0) + n
        else:
            new = dict(target)
            new.update(counts)
        return {c: n for c, n in new.items() if n > 0}

    def set_order(self, counts: dict, mode: str, t: float) -> dict:
        """Apply an order (see compose). Starts a new revision; approvals of over-delivery belong to the old order
        and are dropped."""
        new = self.compose(self.target, counts, mode)
        counts = {str(c): int(n) for c, n in (counts or {}).items()}
        self.revision += 1
        self.target = new
        self.accepted_extra = {}
        return self._event(t, "order", mode=mode, counts=counts)

    def verified(self, color: str, delta: int, t: float, block: int | None = None) -> dict:
        """A placement (+1) or a retrieval (-1) confirmed by observation."""
        self.delivered[color] = self.delivered.get(color, 0) + int(delta)
        if self.delivered[color] < 0:
            raise ValueError(f"delivered {color} below zero")
        return self._event(t, "verified", color=color, delta=int(delta), block=block)

    def accept_extra(self, color: str, n: int, t: float) -> dict:
        self.accepted_extra[color] = self.accepted_extra.get(color, 0) + int(n)
        return self._event(t, "accept_extra", color=color, n=int(n))

    def mark_uncertain(self, color: str, t: float, block: int | None = None, reason: str = "") -> dict:
        self.uncertain[color] = self.uncertain.get(color, 0) + 1
        return self._event(t, "uncertain", color=color, block=block, reason=reason)

    # ---- replay ------------------------------------------------------------------------
    @classmethod
    def replay(cls, events) -> "Ledger":
        """Rebuild a ledger from its events (a log); every event's recorded snapshot must match."""
        led = cls()
        for e in events:
            k = e["kind"]
            if k == "order":
                led.set_order(e["counts"], e["mode"], e["t"])
            elif k == "verified":
                led.verified(e["color"], e["delta"], e["t"], e.get("block"))
            elif k == "accept_extra":
                led.accept_extra(e["color"], e["n"], e["t"])
            elif k == "uncertain":
                led.mark_uncertain(e["color"], e["t"], e.get("block"), e.get("reason", ""))
            else:
                continue
            if "ledger" in e and led.snapshot() != e["ledger"]:
                raise ValueError(f"replay mismatch at {e}: {led.snapshot()}")
        return led

    def copy(self) -> "Ledger":
        return copy.deepcopy(self)
