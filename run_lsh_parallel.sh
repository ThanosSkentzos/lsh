#!/usr/bin/env bash

# Remove old results file if it exists
# rm -f results_lsh.csv

# Python script will write the header, so no need to write it here
# or if the script doesn't do that, you can write it once:
# echo "band,signature_len,seed,pairs_found,time" >> results_lsh.csv

parallel -j $(($(nproc) - 10)) \
    python lsh.py -b {2} -s {3} {1} \
    ::: {6..10} \
    ::: $(seq 10 1 20) \
    ::: $(seq 60 5 100)

