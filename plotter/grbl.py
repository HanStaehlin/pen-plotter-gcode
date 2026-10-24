"""Talk to a GRBL plotter over USB serial: settings, jogging, streaming.

Written against the AX4's GRBL 0.9i (Arduino, 115200 baud, no homing, pen on
M3/M5), and kept to what 1.1 also understands.

Streaming uses character counting rather than send-and-wait.  GRBL plans
moves from what is in its 127-byte receive buffer; waiting for each ``ok``
before sending the next line leaves that buffer nearly empty, and on
photo-derived paths - thousands of sub-millimetre segments - the planner then
runs dry and the pen stutters.  Counting the bytes in flight keeps it full.

Pause, resume, status and reset are GRBL's single-byte real-time commands,
which bypass the buffer and act immediately.
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
import time
from collections import deque

RX_BUFFER = 127                 # GRBL's serial receive buffer, minus one
STATUS_EVERY_S = 0.25

_STATUS = re.compile(r"<([^,|>]+)[,|](.*)>")
_SETTING = re.compile(r"^\$(\d+)=([-\d.]+)")
_WORD = re.compile(r"([XY])(-?\d+\.?\d*)")


#: Settings read from the machine on connect, so plot-time estimates use the
#: real acceleration and corner tolerance rather than guesses.  Per machine,
#: so not committed.
PROFILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "machine.json")


def save_profile(info: dict, path: str | None = None):
    with open(path or PROFILE, "w") as f:
        json.dump(info, f, indent=2)


def load_profile(path: str | None = None) -> dict | None:
    try:
        with open(path or PROFILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def apply_profile(cfg, profile: dict | None = None):
    """Use the machine's own limits for estimates and keep feeds within them."""
    profile = profile if profile is not None else load_profile()
    if not profile:
        return cfg
    accel = [a for a in (profile.get("accel") or []) if a]
    if accel:
        cfg.machine.accel_mm_s2 = float(min(accel))
    if profile.get("junction_deviation"):
        cfg.machine.junction_deviation = float(profile["junction_deviation"])
    rate = [r for r in (profile.get("max_rate") or []) if r]
    if rate:
        cfg.machine.feed_travel = min(cfg.machine.feed_travel, float(min(rate)))
        cfg.machine.feed_draw = min(cfg.machine.feed_draw, float(min(rate)))
    return cfg


def ports() -> list[str]:
    """Serial devices that could be a plotter (macOS names first)."""
    found = sorted(glob.glob("/dev/cu.*")) + sorted(glob.glob("/dev/ttyUSB*")) \
        + sorted(glob.glob("/dev/ttyACM*"))
    return [p for p in found if "Bluetooth" not in p and "debug" not in p.lower()]


def clean(line: str) -> str:
    """One G-code line as GRBL should receive it: no comments, no blanks.

    Comments cost receive-buffer space and GRBL 0.9 throws them away anyway.
    Parenthesised comments go first: they may contain a semicolon, and
    cutting at it would leave half a comment for GRBL to reject.  They are
    stripped by depth, so a nested one - "(Bettina (frog))" - goes whole
    instead of leaving a stray ")" behind.
    """
    out, depth = [], 0
    for ch in line:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            if ch == ";":
                break
            out.append(ch)
    return "".join(out).strip()


def extents(lines) -> tuple[float, float, float, float] | None:
    """(xmin, xmax, ymin, ymax) of absolute X/Y moves in a G-code program."""
    xs, ys = [], []
    x = y = 0.0
    relative = False
    for line in lines:
        up = line.upper()
        if "G91" in up:
            relative = True
        if "G90" in up:
            relative = False
        if not (up.startswith("G0") or up.startswith("G1")) or relative:
            continue
        for axis, val in _WORD.findall(up):
            if axis == "X":
                x = float(val)
            else:
                y = float(val)
        xs.append(x)
        ys.append(y)
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


class Grbl:
    """One connection to one machine.  Thread-safe; the app shares one."""

    def __init__(self):
        self._serial = None
        self._write_lock = threading.Lock()
        self._lock = threading.Lock()
        self._inflight: deque = deque()     # (bytes, kind, waiter)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.port = None
        self.version = ""
        self.settings: dict[int, float] = {}
        self.state = "Disconnected"
        self.wpos = (0.0, 0.0)
        self.last_error = ""
        self.job: dict | None = None

    # ── connection ──────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._serial is not None

    def connect(self, port: str, baud: int = 115200, serial_factory=None) -> dict:
        """Open the port, wait out the Arduino's reset, read version and settings.

        Opening the port toggles DTR, which resets an Arduino-based GRBL; with
        no homing, wherever the pen sits now becomes the machine origin.
        """
        if self.connected:
            self.disconnect()
        if serial_factory is None:
            import serial
            serial_factory = serial.Serial
        s = serial_factory(port, baud, timeout=0.1)
        self._serial = s
        self.port = port
        self._stop.clear()
        time.sleep(2.0)
        s.write(b"\r\n\r\n")
        time.sleep(0.3)
        banner = self._drain(1.0)
        for ln in banner:
            if ln.startswith("Grbl"):
                self.version = ln.split("[")[0].strip()
        self.state = "Idle"
        self._threads = [threading.Thread(target=self._reader, daemon=True),
                         threading.Thread(target=self._poller, daemon=True)]
        for t in self._threads:
            t.start()
        self.read_settings()
        if not self.version:
            # No start-up banner (the board did not reset): ask.  0.9 answers
            # "[0.9i.20150620:]", 1.1 "[VER:1.1h.20190825:]".
            for ln in self.command("$I", timeout=3.0):
                m = re.match(r"\[(?:VER:)?([\d.]+[a-z]?)", ln)
                if m:
                    self.version = "Grbl " + m.group(1)
        return self.info()

    def disconnect(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self.state = "Disconnected"
        with self._lock:
            for _, _, waiter in self._inflight:
                if waiter is not None:
                    waiter["done"].set()
            self._inflight.clear()

    def info(self) -> dict:
        s = self.settings
        return {
            "port": self.port, "version": self.version or "Grbl",
            "travel": [s.get(130), s.get(131)],
            "max_rate": [s.get(110), s.get(111)],
            "accel": [s.get(120), s.get(121)],
            "junction_deviation": s.get(11),
            "steps_per_mm": [s.get(100), s.get(101)],
        }

    # ── low-level I/O ───────────────────────────────────────────────────

    def _write(self, data: bytes):
        with self._write_lock:
            self._serial.write(data)

    def realtime(self, char: bytes):
        """Pause (!), resume (~), status (?), reset (0x18): bypass the queue."""
        if self.connected:
            self._write(char)

    def _drain(self, seconds: float) -> list[str]:
        """Read whatever arrives for a while (before the reader thread runs)."""
        out, end = [], time.time() + seconds
        while time.time() < end:
            raw = self._serial.readline()
            if raw:
                out.append(raw.decode(errors="replace").strip())
        return out

    def _reader(self):
        while not self._stop.is_set():
            try:
                raw = self._serial.readline()
            except Exception as exc:
                self.last_error = f"serial: {exc}"
                self.state = "Disconnected"
                return
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            m = _STATUS.match(line)
            if m:
                self._status(m.group(1), m.group(2))
                continue
            if line == "ok" or line.startswith("error"):
                self._ack(line)
                continue
            if line.startswith("ALARM"):
                self.state = "Alarm"
                self.last_error = line
                if self.job is not None and self.job["state"] == "running":
                    self.job["errors"].append(line)
                    self.job["state"] = "paused"
                continue
            if line.startswith("Grbl"):
                # A reset: GRBL forgot everything that was in flight.
                self.version = line.split("[")[0].strip()
                with self._lock:
                    for _, _, waiter in self._inflight:
                        if waiter is not None:
                            waiter["lines"].append(line)
                            waiter["done"].set()
                    self._inflight.clear()
                continue
            with self._lock:
                for _, _, waiter in self._inflight:
                    if waiter is not None:
                        waiter["lines"].append(line)
                        break
            sm = _SETTING.match(line)
            if sm:
                self.settings[int(sm.group(1))] = float(sm.group(2))

    def _ack(self, line: str):
        with self._lock:
            if not self._inflight:
                return
            _, kind, waiter = self._inflight.popleft()
        if waiter is not None:
            waiter["lines"].append(line)
            waiter["done"].set()
        if kind == "job" and self.job is not None:
            self.job["acked"] += 1
            if line.startswith("error"):
                self.job["errors"].append(f"line {self.job['acked']}: {line}")

    def _status(self, state: str, fields: str):
        self.state = state.split(":")[0]
        # 0.9 reports "MPos:x,y,z,WPos:x,y,z"; 1.1 reports MPos and, now and
        # then, the work offset "WCO:x,y,z" to subtract from it.
        found = {key: (float(m.group(1)), float(m.group(2)))
                 for key in ("WPos", "MPos", "WCO")
                 for m in [re.search(key + r":(-?[\d.]+),(-?[\d.]+)", fields)] if m}
        if "WCO" in found:
            self._wco = found["WCO"]
        if "WPos" in found:
            self.wpos = found["WPos"]
        elif "MPos" in found:
            wco = getattr(self, "_wco", (0.0, 0.0))
            self.wpos = (found["MPos"][0] - wco[0], found["MPos"][1] - wco[1])

    def _poller(self):
        while not self._stop.is_set():
            try:
                self.realtime(b"?")
            except Exception:
                pass
            time.sleep(STATUS_EVERY_S)

    # ── commands ────────────────────────────────────────────────────────

    def command(self, line: str, timeout: float = 10.0) -> list[str]:
        """Send one line and wait for its ok/error; returns the reply lines."""
        if not self.connected:
            raise RuntimeError("not connected")
        if self.streaming:
            raise RuntimeError("a plot is running")
        data = (clean(line) + "\n").encode()
        waiter = {"done": threading.Event(), "lines": []}
        with self._lock:
            self._inflight.append((len(data), "cmd", waiter))
        self._write(data)
        if not waiter["done"].wait(timeout):
            with self._lock:
                try:
                    self._inflight.remove((len(data), "cmd", waiter))
                except ValueError:
                    pass
            raise TimeoutError(f"no reply to {line!r}")
        reply = waiter["lines"]
        if reply and reply[-1].startswith("error"):
            raise RuntimeError(f"{line} -> {reply[-1]}")
        return reply

    def read_settings(self) -> dict[int, float]:
        self.command("$$", timeout=5.0)
        return self.settings

    def jog(self, dx: float, dy: float, feed: float = 2000.0):
        """Relative pen-up move.  GRBL 0.9 has no $J=, so it is G91 then G90."""
        self.command(f"G91 G0 X{dx:.3f} Y{dy:.3f}")
        self.command("G90")

    def set_origin(self):
        """Make the pen's current position X0 Y0 for the next plot."""
        self.command("G92 X0 Y0")
        self.wpos = (0.0, 0.0)

    def go_origin(self, pen_up: str = "M5"):
        self.command(pen_up)
        self.command("G90 G0 X0 Y0")

    def unlock(self):
        self.command("$X")

    # ── streaming ───────────────────────────────────────────────────────

    @property
    def streaming(self) -> bool:
        return self.job is not None and self.job["state"] in ("running", "paused")

    def start(self, program: str, name: str = "", estimate_s: float = 0.0):
        """Stream a program in the background; progress lands in ``self.job``."""
        if not self.connected:
            raise RuntimeError("not connected")
        if self.streaming:
            raise RuntimeError("a plot is already running")
        lines = [c for c in (clean(ln) for ln in program.splitlines()) if c]
        self.job = {"name": name, "total": len(lines), "sent": 0, "acked": 0,
                    "errors": [], "state": "running", "started": time.time(),
                    "estimate_s": estimate_s, "paused_s": 0.0, "paused_at": None}
        t = threading.Thread(target=self._stream, args=(lines,), daemon=True)
        t.start()

    def _stream(self, lines: list[str]):
        job = self.job
        for line in lines:
            data = (line + "\n").encode()
            while True:
                if job["state"] == "stopped" or not self.connected:
                    return
                with self._lock:
                    used = sum(n for n, _, _ in self._inflight)
                if used + len(data) <= RX_BUFFER and job["state"] == "running":
                    break
                time.sleep(0.002)
            with self._lock:
                self._inflight.append((len(data), "job", None))
            self._write(data)
            job["sent"] += 1
            if job["errors"]:
                # A rejected line means the program and the machine disagree
                # about something; carrying on draws garbage.  Hold and ask.
                self.pause()
                job["state"] = "paused"
        while job["acked"] < job["total"] and job["state"] != "stopped" \
                and self.connected:
            time.sleep(0.05)
        while self.state not in ("Idle", "Disconnected") and job["state"] != "stopped":
            time.sleep(0.1)
        if job["state"] != "stopped":
            job["state"] = "done"
            job["finished"] = time.time()

    def pause(self):
        self.realtime(b"!")
        if self.job and self.job["state"] == "running":
            self.job["state"] = "paused"
            self.job["paused_at"] = time.time()

    def resume(self):
        self.realtime(b"~")
        if self.job and self.job["state"] == "paused":
            if self.job["paused_at"]:
                self.job["paused_s"] += time.time() - self.job["paused_at"]
            self.job["paused_at"] = None
            self.job["state"] = "running"

    def stop(self, pen_up: str = "M5"):
        """Stop now without losing position: hold, reset, lift the pen.

        A reset while moving makes GRBL distrust its position and raise an
        alarm, so the motion is decelerated to a hold first.  The reset then
        clears the queued lines, and the pen goes up.
        """
        if not self.connected:
            return
        if self.job is not None and self.job["state"] in ("running", "paused"):
            self.job["state"] = "stopped"
        self.realtime(b"!")
        end = time.time() + 3.0
        while self.state not in ("Hold", "Idle") and time.time() < end:
            time.sleep(0.05)
        time.sleep(0.2)
        self.realtime(b"\x18")
        time.sleep(1.0)
        with self._lock:
            self._inflight.clear()
        if self.state == "Alarm":
            try:
                self.command("$X", timeout=3.0)
            except Exception:
                pass
        try:
            self.command(pen_up, timeout=3.0)
        except Exception as exc:
            self.last_error = f"could not lift pen after stop: {exc}"

    def progress(self) -> dict:
        job = self.job
        out = {"connected": self.connected, "state": self.state,
               "x": self.wpos[0], "y": self.wpos[1], "port": self.port,
               "error": self.last_error, "job": None}
        if job is not None:
            now = job.get("finished") or time.time()
            paused = job["paused_s"] + (time.time() - job["paused_at"]
                                        if job["paused_at"] else 0.0)
            elapsed = max(0.0, now - job["started"] - paused)
            frac = job["acked"] / max(1, job["total"])
            remaining = None
            if job["estimate_s"] > 0:
                remaining = max(0.0, job["estimate_s"] * (1.0 - frac))
            out["job"] = {"name": job["name"], "state": job["state"],
                          "done": job["acked"], "total": job["total"],
                          "fraction": frac, "elapsed_s": elapsed,
                          "remaining_s": remaining, "errors": job["errors"][-3:]}
        return out
