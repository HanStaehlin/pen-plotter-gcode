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
| car_grille | illustration | 25m 32s | 0.241 | 0.396 | 904 | 23.8 | 6.7 |
| car_grille | crosshatch | 1h 17m | 0.711 | 0.078 | 1819 | 93.6 | 4.9 |
| car_grille | flowlines | 58m 01s | 0.665 | 0.087 | 1283 | 94.7 | 3.4 |
| cat_tabby | illustration | 20m 19s | 0.103 | 0.194 | 463 | 13.9 | 2.2 |
| cat_tabby | crosshatch | 1h 00m | 0.679 | 0.058 | 2548 | 52.9 | 3.2 |
| cat_tabby | flowlines | 57m 01s | 0.648 | 0.062 | 2412 | 52.9 | 3.0 |
| dog_ottawa | illustration | 19m 12s | 0.137 | 0.381 | 507 | 14.8 | 2.2 |
| dog_ottawa | crosshatch | 59m 02s | 0.739 | 0.076 | 1316 | 74.9 | 3.9 |
| dog_ottawa | flowlines | 54m 24s | 0.681 | 0.079 | 1353 | 75.6 | 3.5 |
| landscape_lake | illustration | 19m 31s | 0.111 | 0.463 | 477 | 15.6 | 2.5 |
| landscape_lake | crosshatch | 1h 08m | 0.723 | 0.089 | 1566 | 86.5 | 4.3 |
| landscape_lake | flowlines | 55m 15s | 0.647 | 0.105 | 1263 | 81.8 | 3.3 |
| portrait_beard | illustration | 14m 54s | 0.266 | 0.320 | 383 | 10.9 | 1.9 |
| portrait_beard | crosshatch | 34m 53s | 0.803 | 0.072 | 833 | 38.8 | 2.4 |
| portrait_beard | flowlines | 32m 06s | 0.761 | 0.074 | 834 | 38.6 | 2.3 |
| portrait_elderly | illustration | 9m 51s | 0.225 | 0.405 | 361 | 9.0 | 1.4 |
| portrait_elderly | crosshatch | 32m 13s | 0.800 | 0.091 | 779 | 39.6 | 2.4 |
| portrait_elderly | flowlines | 26m 47s | 0.753 | 0.096 | 744 | 38.1 | 2.1 |
