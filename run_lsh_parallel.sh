#!/usr/bin/env bash

# Remove old results file if it exists
# rm -f results_lsh.csv

# Python script will write the header, so no need to write it here
# or if the script doesn't do that, you can write it once:
# echo "band,signature_len,seed,pairs_found,time" >> results_lsh.csv

parallel -j 4 \
    python main.py -b {2} -s {3} {1} \
    ::: {6..7} \
    ::: $(seq 50 10 100) \
    ::: $(seq 100 100 1000)

