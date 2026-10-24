# Evaluation

The six CC0 photographs in `samples/photos/` run through the three main
engines at default settings. Each sheet shows the source, then
`illustration`, `crosshatch` and `flowlines`, labelled with estimated plot
time, pen lifts and metres of line drawn.

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

| photo | engine | plot time | lifts | drawn m | compute s |
|---|---|---|---|---|---|
| car_grille | illustration | 27m 52s | 1013 | 25.9 | 5.7 |
| car_grille | crosshatch | 1h 11m | 1840 | 85.9 | 4.2 |
| car_grille | flowlines | 58m 01s | 1283 | 94.7 | 3.1 |
| cat_tabby | illustration | 21m 08s | 504 | 14.4 | 2.0 |
| cat_tabby | crosshatch | 58m 31s | 2532 | 51.3 | 2.9 |
| cat_tabby | flowlines | 57m 01s | 2412 | 52.9 | 2.7 |
| dog_ottawa | illustration | 22m 59s | 696 | 17.7 | 2.0 |
| dog_ottawa | crosshatch | 56m 57s | 1342 | 71.5 | 3.3 |
| dog_ottawa | flowlines | 54m 24s | 1353 | 75.6 | 3.0 |
| landscape_lake | illustration | 24m 46s | 796 | 20.0 | 2.1 |
| landscape_lake | crosshatch | 1h 05m | 1569 | 81.7 | 3.7 |
| landscape_lake | flowlines | 55m 15s | 1263 | 81.8 | 3.0 |
| portrait_beard | illustration | 15m 53s | 418 | 11.8 | 1.7 |
| portrait_beard | crosshatch | 33m 23s | 845 | 36.8 | 2.2 |
| portrait_beard | flowlines | 32m 06s | 834 | 38.6 | 2.1 |
| portrait_elderly | illustration | 10m 30s | 378 | 9.6 | 1.3 |
| portrait_elderly | crosshatch | 30m 26s | 804 | 36.8 | 2.2 |
| portrait_elderly | flowlines | 26m 47s | 744 | 38.1 | 1.9 |
