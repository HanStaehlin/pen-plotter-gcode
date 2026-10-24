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
| car_grille | illustration | 25m 04s | 0.233 | 0.399 | 886 | 23.3 | 6.0 |
| car_grille | crosshatch | 1h 17m | 0.715 | 0.079 | 1906 | 93.2 | 5.1 |
| car_grille | flowlines | 57m 52s | 0.665 | 0.088 | 1302 | 94.2 | 3.6 |
| cat_tabby | illustration | 20m 14s | 0.102 | 0.194 | 463 | 13.9 | 2.5 |
| cat_tabby | crosshatch | 1h 00m | 0.679 | 0.059 | 2561 | 53.0 | 3.3 |
| cat_tabby | flowlines | 57m 56s | 0.653 | 0.061 | 2521 | 53.0 | 3.2 |
| dog_ottawa | illustration | 19m 11s | 0.136 | 0.381 | 495 | 14.8 | 2.4 |
| dog_ottawa | crosshatch | 59m 23s | 0.740 | 0.075 | 1324 | 75.1 | 3.8 |
| dog_ottawa | flowlines | 53m 52s | 0.680 | 0.079 | 1285 | 75.6 | 3.6 |
| landscape_lake | illustration | 19m 39s | 0.109 | 0.464 | 503 | 15.5 | 2.9 |
| landscape_lake | crosshatch | 1h 08m | 0.731 | 0.089 | 1567 | 86.4 | 4.3 |
| landscape_lake | flowlines | 55m 15s | 0.644 | 0.104 | 1251 | 81.8 | 3.6 |
| portrait_beard | illustration | 14m 44s | 0.261 | 0.321 | 373 | 10.8 | 2.2 |
| portrait_beard | crosshatch | 34m 40s | 0.804 | 0.071 | 811 | 38.9 | 2.6 |
| portrait_beard | flowlines | 31m 56s | 0.755 | 0.074 | 819 | 38.4 | 2.5 |
| portrait_elderly | illustration | 9m 49s | 0.227 | 0.404 | 348 | 9.1 | 1.7 |
| portrait_elderly | crosshatch | 32m 48s | 0.799 | 0.093 | 811 | 39.7 | 2.7 |
| portrait_elderly | flowlines | 26m 32s | 0.753 | 0.096 | 739 | 38.0 | 2.2 |
