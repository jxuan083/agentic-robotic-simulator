"""Generate a local HTML readback with native images and joint plots."""

from bisect import bisect_left
import json

from agentic_sim.real_control import JOINTS, UNITS


def write_viewer(episode):
    data = {"id": episode.manifest["episode_id"], "source": episode.manifest["source"],
            "joints": JOINTS, "units": UNITS, "arms": list(episode.manifest["arms"]),
            "rows": episode.rows, "camera_names": list(episode.frames), "images": {}}
    for camera, rows in episode.frames.items():
        stamps = [row["capture_ns"] for row in rows]
        selected = []
        for control in episode.rows:
            stamp = control["timestamp_ns"]
            i = bisect_left(stamps, stamp)
            nearest = min((max(0, i - 1), min(i, len(stamps) - 1)), key=lambda j: abs(stamps[j] - stamp))
            chosen = rows[nearest]
            selected.append({"rgb": f"{camera}/{chosen['rgb']}", "offset_ms": (stamps[nearest] - stamp) / 1e6})
        data["images"][camera] = selected
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    path = episode.root / "replay.html"
    path.write_text(_HTML.replace("<!--EPISODE-->", serialized), encoding="utf-8")
    return path


_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 episode readback</title>
<style>
body{font:16px system-ui;max-width:1050px;margin:auto;padding:18px;background:#f5f6f8;color:#162331}
h1{font-size:24px;margin:8px 0} label{margin-right:16px} input[type=range]{width:min(500px,95%)}
.images{display:flex;gap:14px;flex-wrap:wrap} figure{flex:1;min-width:250px;margin:10px 0}
img{width:100%;height:auto;display:block;background:#dfe6ed} figcaption{margin:6px 0}
canvas{display:block;width:100%;height:235px;background:#fff;border:1px solid #d2dae4;margin:12px 0}
table{background:#fff;border-collapse:collapse;width:100%} th,td{padding:5px 12px;text-align:left;border-bottom:1px solid #e3e9f0}
</style>
<h1 id="title"></h1>
<p id="time"></p>
<label>Frame <input id="frame" type="range" min="0" value="0"></label>
<div class="images" id="images"></div>
<label>Arm <select id="arm"></select></label>
<label>Joint <select id="joint"></select></label>
<canvas id="plot" width="1000" height="235" aria-label="Command and measured position over time"></canvas>
<table><thead><tr><th>Joint</th><th>Unit</th><th>Command</th><th>Readback</th></tr></thead><tbody id="values"></tbody></table>
<script type="application/json" id="episode"><!--EPISODE--></script>
<script>
const elem = id => document.getElementById(id);
const d = JSON.parse(elem('episode').textContent), rows = d.rows;
elem('title').textContent = d.id + ' · ' + d.source;
elem('frame').max = rows.length - 1;
for (const name of d.arms) elem('arm').add(new Option(name));
for (const name of d.joints) elem('joint').add(new Option(name));
const figures = {};
for (const name of d.camera_names) {
  const fig = document.createElement('figure'), img = document.createElement('img'), cap = document.createElement('figcaption');
  img.alt = name + ' camera'; fig.append(img, cap); elem('images').append(fig); figures[name] = [img, cap];
}
function draw(track, arm, joint, index) {
  const canvas = elem('plot'), ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height;
  const first = rows[0].timestamp_ns, span = Math.max(1, rows.at(-1).timestamp_ns - first);
  let min = Infinity, max = -Infinity;
  for (const [command, readback] of track) {
    min = Math.min(min, command, readback); max = Math.max(max, command, readback);
  }
  const x = row => 70 + (row.timestamp_ns - first) / span * (W - 86);
  const y = value => H - 36 - (value - min + .05) / Math.max(.1, max - min + .1) * (H - 65);
  ctx.clearRect(0,0,W,H); ctx.font = '14px system-ui'; ctx.fillStyle = '#192a38';
  ctx.fillText(arm + ' ' + joint + ' (' + d.units[joint] + ')', 12, 18);
  ctx.fillText(max.toFixed(2), 9, y(max)); ctx.fillText(min.toFixed(2), 9, y(min));
  ctx.fillText(((rows.at(-1).timestamp_ns - first)/1e9).toFixed(3) + ' s', W-73, H-8);
  ctx.fillText('0 s', 70, H-8);
  const line = (at, color, dashed) => {
    ctx.strokeStyle=color;ctx.setLineDash(dashed ? [5,4] : []);ctx.beginPath();
    ctx.moveTo(at,25);ctx.lineTo(at,H-34);ctx.stroke();ctx.setLineDash([]);
  };
  for (const row of rows) if (row.event?.type === 'takeover') line(x(row), '#a47319', true);
  line(x(rows[index]), '#86939e', false);
  ['#1562ba','#ac431e'].forEach((color, series) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    rows.forEach((r,i) => i ? ctx.lineTo(x(r), y(track[i][series])) : ctx.moveTo(x(r), y(track[i][series])));
    ctx.stroke(); ctx.fillStyle = color; ctx.fillText(series ? 'Readback' : 'Command', W - 207 + series*98, 18);
  });
}
function render() {
  const i = Number(elem('frame').value), row = rows[i], arm = elem('arm').value, joint = elem('joint').value;
  elem('time').textContent = 'Frame ' + i + ' of ' + (rows.length-1) + ' · ' +
    (row.timestamp_ns/1e9).toFixed(3) + ' s · ' + row.command_source +
    (row.event ? ' · takeover (' + row.event.discarded_policy_actions + ' queued actions discarded)' : '');
  for (const name of d.camera_names) {
    const [img, cap] = figures[name], frame = d.images[name][i];
    img.src = frame.rgb; cap.textContent = name + ' · image offset ' + frame.offset_ms.toFixed(1) + ' ms';
  }
  elem('values').replaceChildren();
  for (const name of d.joints) {
    const tr = elem('values').insertRow();
    [name, d.units[name], row.arms[arm].command[name].toFixed(3), row.arms[arm].readback[name].toFixed(3)]
      .forEach(v => tr.insertCell().textContent = v);
  }
  draw(rows.map(r => [r.arms[arm].command[joint], r.arms[arm].readback[joint]]), arm, joint, i);
}
['frame','arm','joint'].forEach(id => elem(id).addEventListener('input',render));
render();
</script>
</html>
"""
