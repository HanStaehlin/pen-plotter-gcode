#!/usr/bin/env python3
"""A pretend GRBL 0.9i on a pseudo-terminal, for testing the sender.

    python3 tools/fake_grbl.py          # prints the device path to connect to

Answers $$, $I, $X, ? ! ~ and Ctrl-X like the AX4's firmware, tracks the pen
position through G90/G91/G92, and - the point of it - fails loudly if a
sender ever puts more than 127 bytes into its receive buffer, which on real
hardware silently corrupts lines.
"""

from __future__ import annotations

import os
import re
import threading
import time
import tty

SETTINGS = {0: 10, 1: 25, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 10: 3, 11: 0.010,
            12: 0.002, 13: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 25, 25: 500,
            26: 250, 27: 1, 100: 80, 101: 80, 102: 80, 110: 6000, 111: 6000,
            112: 500, 120: 600, 121: 600, 122: 10, 130: 300, 131: 210, 132: 200}
RX = 128


class FakeGrbl:
    def __init__(self, seconds_per_line: float = 0.0005):
        self.master, slave = os.openpty()
        tty.setraw(slave)               # no echo: a real UART does not echo
        self.path = os.ttyname(slave)
        self._slave = slave
        self.per_line = seconds_per_line
        self.buffer = bytearray()
        self.queue: list[str] = []
        self.overflow = False
        self.received: list[str] = []
        self.pos = [0.0, 0.0]
        self.offset = [0.0, 0.0]
        self.relative = False
        self.hold = False
        self.state = "Idle"
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._run, daemon=True).start()
        self._send("\r\nGrbl 0.9i ['$' for help]")

    def _send(self, text: str):
        os.write(self.master, (text + "\r\n").encode())

    def _read(self):
        while True:
            try:
                data = os.read(self.master, 1024)
            except OSError:
                return
            for b in data:
                c = bytes([b])
                if c == b"?":
                    wx, wy = (self.pos[0] - self.offset[0], self.pos[1] - self.offset[1])
                    self._send(f"<{self.state},MPos:{self.pos[0]:.3f},{self.pos[1]:.3f},0.000,"
                               f"WPos:{wx:.3f},{wy:.3f},0.000>")
                elif c == b"!":
                    self.hold = True
                    self.state = "Hold"
                elif c == b"~":
                    self.hold = False
                elif c == b"\x18":
                    with self.lock:
                        self.buffer.clear()
                        self.queue.clear()
                    self.hold = False
                    self.state = "Idle"
                    self._send("\r\nGrbl 0.9i ['$' for help]")
                else:
                    with self.lock:
                        self.buffer += c
                        while b"\n" in self.buffer:
                            line, _, rest = bytes(self.buffer).partition(b"\n")
                            self.queue.append(line.decode().strip())
                            self.buffer = bytearray(rest)
                        # A line holds its bytes in the receive buffer until
                        # GRBL has parsed it.
                        used = len(self.buffer) + sum(len(q) + 1 for q in self.queue)
                        if used > RX:
                            self.overflow = True

    def _run(self):
        while True:
            with self.lock:
                line = self.queue[0] if self.queue and not self.hold else None
            if line is None:
                if not self.hold:
                    self.state = "Idle"
                time.sleep(0.001)
                continue
            self.state = "Run"
            time.sleep(self.per_line)
            reply = self._execute(line)
            with self.lock:
                if self.queue and self.queue[0] == line:
                    self.queue.pop(0)
            self.received.append(line)
            for r in reply:
                self._send(r)

    def _execute(self, line: str) -> list[str]:
        up = line.upper()
        if not up:
            return ["ok"]
        if up == "$$":
            return [f"${k}={v}" for k, v in SETTINGS.items()] + ["ok"]
        if up == "$I":
            return ["[0.9i.20150620:]", "ok"]
        if up == "$X":
            return ["[Caution: Unlocked]", "ok"]
        if up.startswith("$"):
            return ["error: Invalid statement"]
        if "G91" in up:
            self.relative = True
        if "G90" in up:
            self.relative = False
        words = dict(re.findall(r"([XY])(-?\d+\.?\d*)", up))
        if "G92" in up:
            for i, a in enumerate("XY"):
                if a in words:
                    self.offset[i] = self.pos[i] - float(words[a])
            return ["ok"]
        if not re.match(r"^(G[0-4]|M[35]|G9[01]|F)", up) and not up.startswith("G4"):
            return ["error: Unsupported statement"]
        for i, a in enumerate("XY"):
            if a in words:
                v = float(words[a])
                self.pos[i] = self.pos[i] + v if self.relative else v + self.offset[i]
        return ["ok"]


if __name__ == "__main__":
    fake = FakeGrbl(seconds_per_line=0.002)
    print(fake.path, flush=True)
    while True:
        time.sleep(1)
        if fake.overflow:
            print("RX BUFFER OVERFLOW", flush=True)
