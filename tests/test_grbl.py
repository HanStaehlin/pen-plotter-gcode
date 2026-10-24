"""The USB sender, against a pretend GRBL 0.9i (tools/fake_grbl.py)."""

import time

import numpy as np
import pytest

from conftest import wait_job
from plotter import gcode, grbl, speedtest
from plotter.config import Config


@pytest.mark.parametrize("line,expected", [
    ("(Bettina (frog))", ""),                         # nested: no stray ")"
    ("(turned 90 deg; turn the paper)", ""),          # ";" inside a comment
    ("G1 X1 (a) Y2 ; rest", "G1 X1  Y2"),
    ("(unclosed", ""),
    ("M3S1000", "M3S1000"),
])
def test_clean(line, expected):
    assert grbl.clean(line) == expected


def test_extents_ignore_relative_moves():
    prog = ["G90", "G0 X1 Y-2", "G1 X5 Y-9", "G91", "G0 X100", "G90", "G1 X3 Y-1"]
    assert grbl.extents(prog) == (1.0, 5.0, -9.0, -1.0)


def test_connect_reads_version_and_settings(machine):
    info = machine.info()
    assert info["version"] == "Grbl 0.9i"
    assert info["travel"] == [300.0, 210.0]
    assert info["accel"] == [600.0, 600.0]


def test_jog_and_origin(machine, fake):
    machine.jog(10, -5)
    time.sleep(0.4)
    assert machine.wpos == pytest.approx((10, -5))
    machine.set_origin()
    machine.jog(-3, 0)
    machine.go_origin()
    time.sleep(0.4)
    assert machine.wpos == pytest.approx((0, 0))
    assert fake.pos == pytest.approx([10, -5])     # origin moved, machine did not


def _program(n=300, header="Test (frog)"):
    rng = np.random.default_rng(1)
    strokes = [np.cumsum(rng.normal(0, 1, (8, 2)), axis=0) + [50, -50] for _ in range(n)]
    return gcode.write(strokes, Config(), header)


def test_stream_delivers_every_line_in_order_without_overflow(machine, fake):
    prog = _program()
    lines = [c for c in (grbl.clean(ln) for ln in prog.splitlines()) if c]
    machine.start(prog, "test")
    job = wait_job(machine)
    assert job["state"] == "done" and not job["errors"]
    assert job["acked"] == len(lines)
    assert fake.received[-len(lines):] == lines
    assert not fake.overflow                       # never more than 127 bytes in flight


def test_pause_holds_and_resume_finishes(machine):
    fake_slow = machine                            # same connection, just slower lines
    machine.start(_program(400), "test")
    time.sleep(0.2)
    machine.pause()
    done = machine.job["acked"]
    time.sleep(0.5)
    assert machine.job["state"] == "paused" and machine.state == "Hold"
    assert machine.job["acked"] - done <= 2        # only what was mid-parse
    machine.resume()
    assert wait_job(fake_slow)["state"] == "done"


def test_stop_lifts_the_pen(machine, fake):
    machine.start(_program(2000), "test")
    time.sleep(0.2)
    machine.stop()
    assert machine.job["state"] == "stopped"
    assert fake.received[-1] == Config().machine.pen_up_cmd


def test_a_rejected_line_pauses_the_plot(machine):
    machine.start("G90\nG0 X1 Y-1\nG99 nonsense\nG0 X2 Y-2\n", "bad")
    time.sleep(0.5)
    assert machine.job["state"] == "paused"
    assert machine.job["errors"]


def test_speed_test_tries_each_level_and_restores_settings(machine, fake):
    from fake_grbl import SETTINGS
    seen = set()
    blocks = speedtest.build(Config(), levels=((1000, 0.02), (2000, 0.06)))
    machine.run_test(blocks)
    while machine.job["state"] == "running":
        seen.add(SETTINGS[120])
        time.sleep(0.005)
    assert machine.job["state"] == "done" and not machine.job["errors"]
    assert {1000.0, 2000.0} <= seen
    assert (SETTINGS[120], SETTINGS[121], SETTINGS[11]) == (600, 600, 0.010)
    assert len(machine.job["results"]) == sum(len(b["parts"]) for b in blocks)
    assert not fake.overflow


def test_profile_drives_the_estimate():
    cfg = grbl.apply_profile(Config(), {"accel": [1500, 1500], "junction_deviation": 0.04,
                                        "max_rate": [4000, 4000], "lift_overhead_s": 0.05})
    m = cfg.machine
    assert (m.accel_mm_s2, m.junction_deviation, m.lift_overhead_s) == (1500, 0.04, 0.05)
    assert m.feed_travel == 4000                   # clamped to the machine's maximum


def test_speed_test_sheet_stays_on_the_bed():
    for block in speedtest.build(Config()):
        for part in block["parts"]:
            ext = grbl.extents(part["program"].splitlines())
            if ext:
                x0, x1, y0, y1 = ext
                assert 0 <= x0 and x1 <= 300 and -210 <= y0 and y1 <= 0


def test_speed_test_summary_measures_the_lift_cost():
    results = [{"accel": 600, "junction_deviation": 0.01, "name": "dashes", "lifts": 50,
                "estimate_s": 20.0, "actual_s": 24.0},
               {"accel": 600, "junction_deviation": 0.01, "name": "maze", "lifts": 1,
                "estimate_s": 10.0, "actual_s": 11.0}]
    s = speedtest.summarise(results)
    assert s["lift_overhead_s"] == pytest.approx(0.08)
    assert s["levels"][0]["actual_s"] == pytest.approx(35.0)


def test_finished_plots_are_logged_for_calibration(machine, tmp_path):
    import json
    machine.start(_program(50), "logged", estimate_s=12.5)
    assert wait_job(machine)["state"] == "done"
    entry = json.loads((tmp_path / "plot_log.jsonl").read_text().splitlines()[-1])
    assert entry["name"] == "logged" and entry["estimate_s"] == 12.5
    assert entry["actual_s"] > 0 and entry["lines"] == machine.job["total"]
