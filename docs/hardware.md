# Machine and plotting notes

## The AX4 plotter

- GRBL 0.9i firmware (2015 build), CH340 USB-serial chip, **115200 baud**,
  shows up as `/dev/cu.usbserial-110`. macOS drives the CH340 natively; the
  Windows driver from the vendor bundle is not needed.
- Bed 300 × 210 mm, 80 steps/mm, max 6000 mm/min, acceleration 600 mm/s².
  **No homing, no limit switches, no soft limits**: the origin is wherever
  the pen is when the board starts, and nothing stops it running into the
  frame.
- The vendor manual, firmware, drivers and Candle sender are not part of
  this repository; keep them alongside it (locally in
  `~/Documents/AX4 Plotter Resources/`).
- Generated G-code uses `M3S1000` / `M5` for pen down / up and a negative-Y
  coordinate convention, matching the vendor's own files.

## Pen

- **Pen width is the important setting.** It is what calibrates tone. Measure
  the real ink trace, not what the barrel claims.
- **Finer hatching means a finer pen**, and the app needs telling. Tone is
  reproduced by spacing strokes `pen / coverage` apart, so halving the nib
  roughly doubles the line drawn: on one portrait, 0.40 mm gave 24 m and 18
  minutes, 0.25 mm gave 38 m and 25 minutes, 0.15 mm gave 58 m and 40 minutes.
  Tone accuracy holds throughout. A finer pen also affords more hatch
  directions, since each contributes less coverage. The working resolution
  follows the pen automatically when it would otherwise land sub-pixel.

## Plot speed

Measured on this project's output, almost no segment ever reaches the feed
rate: reaching 3000 mm/min at 400 mm/s² takes 3.1 mm, and median segments
are 0.4-2.4 mm. Feed rate is therefore not the limit; acceleration and
corner speed are.

| change | crosshatch | hilbert |
|---|---|---|
| feed 3000 → 6000 mm/min | 1.11× faster | 1.00× |
| acceleration 400 → 1600 mm/s² (`$120`/`$121`) | 1.31× | 1.93× |
| junction deviation 0.02 → 0.10 mm (`$11`) | 1.06× | 1.50× |
| all of the above + `--simplify 0.20` | 1.35× | 2.50× |

(Measured from a 400 mm/s² baseline; the AX4 actually ships at 600.)

Raise `$120/$121` and `$11` only as far as the machine draws cleanly.
Connecting from the app reads them back into `machine.json`, so plot-time
estimates follow automatically.

## Plotting from the app

The app streams G-code to the plotter itself; UGS is not needed. Quit UGS
first, since only one program can hold the serial port.

1. **Connect.** Pick the `usbserial` port under *Plotter*, below the preview.
   Opening the port restarts the Arduino. The app reads the machine's
   settings, so plot-time estimates use the real acceleration and corner
   tolerance from then on.
2. **Set the origin.** Jog the pen to the paper corner the app names and
   press *Set origin here*. For landscape drawings that is the top-left
   corner. A portrait drawing goes to the bed turned a quarter, so its origin
   is the drawing's top-right corner. The red ring on the preview shows
   where the pen is relative to the drawing.
3. **Trace outline.** The pen goes round the drawing's bounding box, pen up.
   If that leaves the paper, move the paper or the origin.
4. **Plot.** Pause holds the motion and can be resumed. Stop decelerates,
   resets GRBL's queue and lifts the pen, without losing position. A line
   GRBL rejects pauses the plot rather than drawing on blindly. A drawing
   that would run past the bed from the origin asks for confirmation first.

Streaming counts characters to keep GRBL's 127-byte receive buffer full
instead of waiting for each `ok`; on photo-derived paths, thousands of short
segments, send-and-wait lets the planner run dry and the pen stutters.
GRBL 0.9 has no `$J=` jog command, so jogging is `G91 G0` then `G90`, and
the origin is a `G92` offset (it does not survive a power cycle).

`tools/fake_grbl.py` is a pretend GRBL 0.9i on a pseudo-terminal for testing
all of this without the machine. It fails loudly if a sender overflows the
receive buffer.

## Portrait plots

A plotter bed that is wider than it is tall cannot draw a portrait page, so
portrait output is turned 90 degrees anticlockwise before it is written; turn
the paper the same way and the drawing comes out upright. The preview keeps
showing the picture as composed — only what goes down the wire is turned.
`--rotate-output off` disables it.

To turn a G-code file that already exists:

```
python3 tools/gcode_rotate.py drawing.gcode          # -> drawing_landscape.gcode
```

This is a rotation, not a transpose. Swapping X and Y reflects the drawing
about the diagonal and comes out mirrored — readable enough on an abstract
pattern that the mistake reaches the paper, and obvious the moment there is
lettering or a face in the picture. A rotation swaps the axes *and* flips the
sign of one of them.
