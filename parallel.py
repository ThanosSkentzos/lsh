# %%
# READ DATA
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
import random
from time import perf_counter
from multiprocessing import Lock

ncols = 80
update_interval = 1
global skipped
skipped = 0
from os.path import exists

goto_prevline = "\033[A"
erase_line = "\r\33[2K"


# %%
def load_data(limit=-1):
    with open("user_movie_rating.npy", "rb") as f:
        data = np.load(f)
    print(f"shape: {data.shape}")
    print(f"removing ratings")
    data = data[:, :2]
    data = data[:limit]
    print(f"shape: {data.shape}")
    return data


def process_data(data):
    print("Aggregating movies for users...")
    df = pd.DataFrame(data)
    df = df.groupby(0).aggregate(list)
    df.columns = ["movies"]
    users = df.index
    movies = np.unique(data[:, 1])
    movies.sort()
    print("Done.")
    print(f"We have {len(users)} users and {len(movies)} movies")
    return df, users, movies


def get_data():
    import pickle

    if exists("data.pickle"):
        try:
            with open("data.pickle", "rb") as f:
                df, users, movies = pickle.load(f)
                return df,users,movies
        except(EOFError,pickle.UnpicklingError):
            print("Error reading data, generating from stratch...") 
    data = load_data()
    df, users, movies = process_data(data)
    with open("data.pickle", "wb") as f:
        pickle.dump((df, users, movies), f)
    return df, users, movies


def jac_sim(li, lj):
    """
    Jaccard similarity assuming list inputs of movie ids, not entire vector
    """
    movie_set_i, movie_set_j = set(li), set(lj)
    intersection = movie_set_i & movie_set_j
    union = movie_set_i | movie_set_j
    if len(union) == 0:
        print(f"found empty union for {li} and {lj}")
        return 0
    return len(intersection) / len(union)


def test_jaccard_similarity():
    # test jaccard similarity
    df = process_data(load_data())
    li = df["movies"].values[36959]
    lj = df["movies"].values[81962]
    print(jac_sim(li, lj))


def one_hot(row, mapping):
    row_hot = np.zeros(len(mapping))
    for m in row:
        try:
            row_hot[mapping[m]] = 1
        except Exception as e:
            print("here")
    return row_hot


# PERMUTATIONS
def calculate_perms(indexes, movies, num_perms=100):
    print(f"Calculating {num_perms} permutations for {len(movies)} movies", end="")
    perms = [np.random.permutation(indexes) for _ in range(num_perms)]
    perm_movie_idx = [
        {int(m): int(idx) for m, idx in zip(movies, perm)} for perm in perms
    ]
    print(f" - Done")
    return perm_movie_idx


def generate_minhash(row, vocab):
    def minhash(perm):
        for m in vocab:
            idx = perm[m]
            sig_val = row[idx]
            if sig_val == 1:
                return m

    return minhash


def minhash_perms(row, perms, vocab):
    mh = generate_minhash(row, vocab)
    sig = list(map(mh, perms))
    return np.array(sig)


def calculate_minhashes(values, perms, vocab):
    sigs = []
    desc = f"Calculating minhashes".ljust(ncols // 3 -7)
    for row in tqdm(
        values, desc=desc, position=0, ncols=ncols, mininterval=update_interval
    ):
        row_hot = one_hot(row, vocab)
        sigs.append(minhash_perms(row_hot, perms, vocab))
    return sigs


file_path = "sigs.pickle"


def save_sigs(sigs):
    import pickle

    with open(file_path, "wb") as f:
        pickle.dump(sigs, f)


def load_sigs(file_path):
    import pickle

    with open(file_path, "rb") as f:
        sigs = pickle.load(f)
    return sigs


def split_sig(sig, b=10):
    splits = np.array_split(sig, b)
    return splits


def calculate_hashes(sigs, b=10):
    hashes = {}
    desc = f"Calculating hashes for b={b}".ljust(ncols // 3 -7)
    for user_idx, sig in enumerate(
        tqdm(
            sigs,
            desc=desc,
            position=0,
            ncols=ncols,
            leave=True,
            mininterval=update_interval,
        )
    ):
        hashes_list = split_sig(sig, b)
        for hash_arr in hashes_list:
            hash = ",".join([f"{i}" for i in sorted(list(set(hash_arr)))])
            hashes.setdefault(hash, set()).add(user_idx+1)
    candidate_sets = [v for v in hashes.values() if len(v) > 1]
    return candidate_sets


def evaluate_candidates(candidate_sets, threshold, values, tried=set()):
    from itertools import combinations
    start = perf_counter()
    stop=False
    lock = Lock()
    max_val = 0
    similars = set()
    global skipped
    desc = "Evaluating candidates".ljust(ncols // 3)
    pbar = tqdm(
        candidate_sets,
        desc=desc,
        position=0,
        ncols=ncols,
        leave=True,
        mininterval=update_interval,
    )
    for cs in pbar:
        pbar.set_postfix({"found": len(similars)})
        if stop:
            break
        if len(cs) > 100:
            continue
            cs = random.sample(list(cs), 100)
        pairs = combinations(cs, 2)
        for pair in pairs:
            now = perf_counter()
            if now-start>1000:
                stop=True
            if stop:
                break
            pair = tuple(sorted(pair))
            if pair in tried:
                skipped += 1
                continue
            if pair in similars:
                continue
            tried.add(pair)
            data1 = values[pair[0]-1]
            data2 = values[pair[1]-1]
            sim = jac_sim(data1, data2)
            if sim > max_val:
                max_val = sim
            if sim > threshold:
                # similars.add(tuple([*pair, sim]))
                similars.add(tuple(pair))
    with lock:
        save_new_pairs("results.txt", similars)
    return similars


def print_header(argument_name):
    """
    Prints a stylized header for a new argument.
    """
    n = ncols - len(argument_name)
    border = "=" * ncols
    print("\n" + border)
    print(f"{(n//2)*' '}{argument_name.upper()}{(n//2)*' '}")
    print(border + "\n")


def get_saved_pairs(file_path):
    if not exists(file_path):
        with open(file_path, "w"):
            pass
    with open(file_path, "r") as f:
        saved_pairs = set(
            [tuple(map(int, i.strip().split(","))) for i in f.readlines() if i.strip()]
        )
    return saved_pairs


def save_new_pairs(file_path, new_pairs):
    saved_pairs = get_saved_pairs(file_path)

    new = new_pairs - saved_pairs
    all_pairs = new_pairs | saved_pairs

    with open(file_path, "w") as f:
        for pair in sorted(all_pairs):
            f.write(f"{pair[0]},{pair[1]}\n")
    return len(new), len(all_pairs)


def parse_args():
    # Default values
    scale = 4
    b = 15 * scale
    signature_len = 100*scale
    n_bands = 1
    random_state = 1 

    args = sys.argv[1:]
    # Simple argument parsing
    # Allowed forms:
    # python script.py -b 20 -s 200 -n 2 123
    # where 123 is the random_state

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-b":
            i += 1
            if i < len(args):
                b = int(args[i])
        elif arg == "-s":
            i += 1
            if i < len(args):
                signature_len = int(args[i])
        elif arg == "-n":
            i += 1
            if i < len(args):
                n_bands = int(args[i])
        else:
            # If it's not a flag, assume it's the random_state
            try:
                random_state = int(arg)
            except ValueError:
                print(
                    f"Warning: Could not parse {arg} as integer. Using default seed={random_state}"
                )
        i += 1

    np.random.seed(random_state)
    random.seed(random_state)
    return b, signature_len, n_bands, random_state


def main():

    b, signature_len, n_bands, random_state = parse_args()
    t0 = perf_counter()

    # print_header(f"b={b}, signature_len={signature_len}, n_bands={n_bands}, seed={random_state}")
    print_header(f"b={b}, signature_len={signature_len},seed={random_state}")
    threshold = 0.5
    df, users, movies = get_data()
    values = df["movies"].values

    movie_indexes = {int(m): np.argmax(movies == m) for m in movies}
    indexes = np.arange(len(movie_indexes))

    # PERMUTATIONS
    perm_movie_idx = calculate_perms(indexes, movies, signature_len)
    # MIN HASHING
    sigs = calculate_minhashes(values=values, perms=perm_movie_idx, vocab=movie_indexes)
    tried = set()
    bmax = b + n_bands

    for band_index in range(b, bmax):
        candidate_sets = calculate_hashes(sigs, band_index)
        similars = evaluate_candidates(candidate_sets, threshold, values, tried)
        t = perf_counter()
        new_pairs = set([i[:2] for i in similars])

        # Remove the fancy printing lines
        # Instead, write a clean CSV line:
        with open("results_lsh.csv", "a") as f:
            # f.write(f"{band_index},{signature_len},{random_state},{len(new_pairs|graph_pairs)},{t - t0:.1f}\n")
            f.write(
                f"{band_index},{signature_len},{random_state},{len(new_pairs)},{t - t0:.1f}\n"
            )

        # if num_all > 1200:
        #     return 0

    t = perf_counter()
    print(f"Finished in {t - t0:.2f} seconds")


if __name__ == "__main__":
    main()
# %%
