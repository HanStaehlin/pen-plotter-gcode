# Evaluation

The six CC0 photographs in `samples/photos/` run through the three main
engines at default settings. Each sheet shows the source, then
`illustration`, `crosshatch` and `flowlines`, labelled with estimated plot
time, likeness and pen lifts.

**Likeness** is SSIM between the finished plot and the photo's tone target,
both seen from arm's length: every G-code path is drawn at the pen's real
width, then both images are blurred by 1 mm (`plotter/quality.py`). Read it
next to plot time to see what a change buys per minute. Compare it within an
engine, not across engines: `illustration` deliberately leaves light and mid
tones to the paper, so it scores far below the dense tone engines even where
it looks better.

Regenerate with:

    python3 tools/evaluate.py -e illustration crosshatch flowlines -o eval_out

Judge changes on the whole set. Two earlier defaults in this project were
wrong because they were tuned on a single picture.

## portrait beard

![portrait_beard](portrait_beard.jpg)

## portrait elderly

![portrait_elderly](portrait_elderly.jpg)

## dog ottawa

![dog_ottawa](dog_ottawa.jpg)

## cat tabby

![cat_tabby](cat_tabby.jpg)

## car grille

![car_grille](car_grille.jpg)

## landscape lake

![landscape_lake](landscape_lake.jpg)

## Numbers

| photo | engine | plot time | likeness | tone error | lifts | drawn m | compute s |
|---|---|---|---|---|---|---|---|
| car_grille | illustration | 25m 59s | 0.194 | 0.423 | 1431 | 19.2 | 6.0 |
| car_grille | crosshatch | 57m 41s | 0.619 | 0.070 | 1156 | 71.1 | 4.4 |
| car_grille | flowlines | 44m 25s | 0.599 | 0.083 | 977 | 71.9 | 3.4 |
| cat_tabby | illustration | 10m 57s | 0.075 | 0.227 | 545 | 5.6 | 1.8 |
| cat_tabby | crosshatch | 27m 46s | 0.479 | 0.071 | 1590 | 19.9 | 2.2 |
| cat_tabby | flowlines | 27m 35s | 0.469 | 0.073 | 1540 | 20.4 | 2.2 |
| dog_ottawa | illustration | 13m 56s | 0.101 | 0.408 | 708 | 8.5 | 2.1 |
| dog_ottawa | crosshatch | 34m 14s | 0.595 | 0.060 | 1001 | 41.4 | 2.9 |
| dog_ottawa | flowlines | 32m 32s | 0.575 | 0.062 | 1041 | 41.4 | 3.0 |
| landscape_lake | illustration | 14m 55s | 0.087 | 0.491 | 742 | 9.7 | 2.2 |
| landscape_lake | crosshatch | 31m 47s | 0.549 | 0.076 | 861 | 38.4 | 2.8 |
| landscape_lake | flowlines | 27m 55s | 0.542 | 0.078 | 861 | 36.0 | 2.5 |
| portrait_beard | illustration | 13m 06s | 0.238 | 0.344 | 625 | 8.1 | 2.1 |
| portrait_beard | crosshatch | 27m 10s | 0.728 | 0.071 | 595 | 31.6 | 2.4 |
| portrait_beard | flowlines | 24m 29s | 0.688 | 0.071 | 565 | 30.9 | 2.4 |
| portrait_elderly | illustration | 7m 09s | 0.126 | 0.458 | 382 | 5.0 | 1.7 |
| portrait_elderly | crosshatch | 16m 51s | 0.620 | 0.082 | 614 | 18.7 | 1.8 |
| portrait_elderly | flowlines | 15m 38s | 0.594 | 0.088 | 619 | 18.8 | 1.8 |
