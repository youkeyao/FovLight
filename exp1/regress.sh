#!/bin/bash

python regress.py --csv_path /mnt/data/youkeyao/FovLight/exp1/results/roughness_0_scores.csv --mat_type 0
python regress.py --csv_path /mnt/data/youkeyao/FovLight/exp1/results/roughness_0p1_scores.csv --mat_type 0.1
python regress.py --csv_path /mnt/data/youkeyao/FovLight/exp1/results/roughness_0p4_scores.csv --mat_type 0.4
python regress.py --csv_path /mnt/data/youkeyao/FovLight/exp1/results/roughness_1_scores.csv --mat_type 1
python regress.py --csv_path /mnt/data/youkeyao/FovLight/exp1/results/metal_scores.csv --mat_type 01