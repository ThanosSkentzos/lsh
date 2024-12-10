import matplotlib.pyplot as plt
import numpy as np
from line_profiler import profile
from tqdm import tqdm
import pandas as pd
import sys
import random
import networkx as nx
from collections import Counter, defaultdict
from os.path import exists

# Define global variables
ncols = 100
update_interval = 1
global skipped
skipped = 0

goto_prevline = "\033[A"
erase_line = "\r\33[2K"

# --- Existing Functions ---

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
        with open("data.pickle", "rb") as f:
            df, users, movies = pickle.load(f)
    else:
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


def verify_jaccard_similarity(file_path, values, jac_sim, threshold=0.5):
    # Read pairs from file
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Parse pairs
    pairs = []
    for line in lines:
        line = line.strip()
        if line:
            u, v = map(int, line.split(","))
            pairs.append((u, v))

    # 1. Check that Jaccard Similarity > 0.5 for all pairs
    all_above_threshold = True
    for u, v in pairs:
        sim = jac_sim(values[u], values[v])
        if sim <= threshold:
            all_above_threshold = False
            print(f"Pair ({u}, {v}) has similarity {sim} which is NOT above {threshold}.")
            break

    if all_above_threshold:
        print("All pairs have Jaccard similarity above 0.5.")

    # 2. Confirm that all pairs are unique and count them
    unique_pairs = set(pairs)
    if len(unique_pairs) == len(pairs):
        print("All pairs are unique. No duplicates found.")
    else:
        print("Duplicates found in pairs.")
    print(f"Total number of unique pairs: {len(unique_pairs)}")

    # If you expect exactly 1217 pairs as stated:
    if len(unique_pairs) == 1217:
        print("Confirmed that we have exactly 1217 different pairs.")

    # 3. Determine how many users are involved and plot the degree scatter plot
    # Build a graph representation
    user_degrees = defaultdict(int)
    users_involved = set()
    for u, v in unique_pairs:
        user_degrees[u] += 1
        user_degrees[v] += 1
        users_involved.add(u)
        users_involved.add(v)

    print(f"Number of unique users involved: {len(users_involved)}")

    # Plot degree scatter plot
    # We'll plot user index vs. degree or simply a scatter of (user_id, degree).
    # Note: user indices might be large; consider plotting them sorted by degree for clarity.
    user_ids = sorted(user_degrees.keys())
    degrees = [user_degrees[uid] for uid in user_ids]

    plt.figure(figsize=(10, 6))
    plt.scatter(user_ids, degrees, alpha=0.5)
    plt.title("User Degree Scatter Plot")
    plt.xlabel("User ID")
    plt.ylabel("Degree (Number of Similar Neighbors)")
    plt.grid(True)
    # plt.show()

    return unique_pairs  # Return unique_pairs for further analysis


def main():
    # Example usage:
    # Make sure that you have `values`, `jac_sim`, and the `results.txt` file in the correct directory.
    df, users, movies = get_data()
    values = df["movies"].values
    unique_pairs = verify_jaccard_similarity("results.txt", values, jac_sim, threshold=0.5)

    # Read the pairs from file and build the graph
    file_path = "results.txt"
    with open(file_path, "r") as f:
        lines = f.readlines()

    pairs = []
    for line in lines:
        line = line.strip()
        if line:
            u, v = map(int, line.split(","))
            pairs.append((u, v))

    # Create an undirected graph
    G = nx.Graph()
    G.add_edges_from(pairs)

    # Basic graph info
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {avg_degree:.4f}")

    # Average clustering
    avg_clustering = nx.average_clustering(G)
    print(f"Average clustering coefficient: {avg_clustering:.4f}")

    # Diameter: We must check connectivity first. Diameter is defined for the largest connected component.
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        print(f"Diameter: {diameter}")
    else:
        # If not connected, find the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc).copy()
        diameter = nx.diameter(G_lcc)
        print(f"Diameter of the largest connected component: {diameter}")
        print("Note: Graph is not fully connected.")

    # Visualize the entire graph with a spring layout
    # Warning: If the graph is large, this can be slow or cluttered.
    if num_nodes > 0:
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_size=30, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.title("Graph Visualization (Spring Layout)")
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
    else:
        print("Graph has no nodes to visualize.")

    # Degree distribution analysis
    degrees = [deg for _, deg in G.degree()]
    degree_count = Counter(degrees)
    if degree_count:
        deg, freq = zip(*degree_count.items())
    else:
        deg, freq = ([], [])

    # Plot degree distribution on a log-log scale
    if deg and freq:
        plt.figure(figsize=(8, 6))
        plt.scatter(deg, freq, alpha=0.7, edgecolors='none')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Degree Distribution on Log-Log Scale")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()
    else:
        print("No degree information available to plot.")

    # --- Additions Start Here ---

    # 1. Average Number of Movies Viewed by All Users
    total_movies_all_users = df["movies"].apply(len).sum()
    num_total_users = len(df)
    average_movies_all_users = total_movies_all_users / num_total_users if num_total_users > 0 else 0
    print(f"Average number of movies viewed by all users: {average_movies_all_users:.2f}")

    # 2. Average Number of Movies Viewed by Users Involved in Pairs with JS > 0.5
    # Extract unique user IDs from unique_pairs
    users_in_pairs = set()
    for u, v in unique_pairs:
        users_in_pairs.add(u)
        users_in_pairs.add(v)
    num_users_in_pairs = len(users_in_pairs)

    # Convert the set to a list before using it as an indexer
    users_in_pairs_list = list(users_in_pairs)

    # Calculate total and average number of movies viewed by these users
    if users_in_pairs_list:
        total_movies_in_pairs = df.loc[users_in_pairs_list, "movies"].apply(len).sum()
        average_movies_in_pairs = total_movies_in_pairs / num_users_in_pairs if num_users_in_pairs > 0 else 0
    else:
        total_movies_in_pairs = 0
        average_movies_in_pairs = 0
    print(f"Average number of movies viewed by users in pairs with Jaccard Similarity > 0.5: {average_movies_in_pairs:.2f}")

    # 3. Number of Movies Viewed by the Top 5 Users
    # Get degrees of all users
    degree_dict = dict(G.degree())
    # Sort users by degree in descending order
    top_5_users = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 Users by Degree and Their Number of Viewed Movies:")
    for rank, (user, degree) in enumerate(top_5_users, start=1):
        # Ensure the user exists in the DataFrame
        if user in df.index:
            num_movies = len(df.at[user, "movies"])
            print(f"{rank}. User {user} - Degree: {degree}, Number of Movies Viewed: {num_movies}")
        else:
            print(f"{rank}. User {user} - Degree: {degree}, Number of Movies Viewed: Data Not Available")

    # 4. Number of Movies for `user_pair_max_sim`
    # Identify the pair with the highest Jaccard similarity
    print("Identifying the pair with the highest Jaccard similarity...")

    # Initialize variables to track the highest similarity
    max_similarity = -1
    user_pair_max_sim = (None, None)

    # Iterate through unique_pairs to find the pair with the highest similarity
    for u, v in unique_pairs:
        sim = jac_sim(values[u], values[v])
        if sim > max_similarity:
            max_similarity = sim
            user_pair_max_sim = (u, v)

    if user_pair_max_sim != (None, None):
        print(f"Highest Jaccard similarity is {max_similarity:.4f} between users {user_pair_max_sim[0]} and {user_pair_max_sim[1]}")

        # Retrieve the number of movies each user in the pair has viewed
        user1, user2 = user_pair_max_sim
        if user1 in df.index:
            num_movies_user1 = len(df.at[user1, "movies"])
        else:
            num_movies_user1 = "Data Not Available"

        if user2 in df.index:
            num_movies_user2 = len(df.at[user2, "movies"])
        else:
            num_movies_user2 = "Data Not Available"

        print(f"Number of movies viewed by User {user1}: {num_movies_user1}")
        print(f"Number of movies viewed by User {user2}: {num_movies_user2}")

        # Optional: Number of common movies between the two users
        if isinstance(num_movies_user1, int) and isinstance(num_movies_user2, int):
            movies_user1 = set(df.at[user1, "movies"])
            movies_user2 = set(df.at[user2, "movies"])
            common_movies = movies_user1 & movies_user2
            num_common_movies = len(common_movies)
            print(f"Number of common movies between User {user1} and User {user2}: {num_common_movies}")
    else:
        print("No user pairs found to determine `user_pair_max_sim`.")

    # 5. Plot the Largest Weakly Connected Component (WCC) Separately
    # Ensure that the graph is not empty
    if G.number_of_nodes() == 0:
        print("The graph has no nodes to plot.")
    else:
        # Identify the largest connected component
        if nx.is_connected(G):
            largest_cc = G
            print("The entire graph is connected. Plotting the entire graph as the Largest Connected Component.")
        else:
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            largest_cc = G.subgraph(largest_cc_nodes).copy()
            print(f"Size of the Largest Connected Component: {largest_cc.number_of_nodes()} users")

        # Compute degrees for node sizing
        degrees_lcc = dict(largest_cc.degree())

        # Compute PageRank for node coloring
        pagerank_lcc = nx.pagerank(largest_cc)

        # Normalize PageRank for color mapping
        pagerank_values = list(pagerank_lcc.values())
        pagerank_min = min(pagerank_values)
        pagerank_max = max(pagerank_values)
        # Handle case where all PageRank values are equal
        if pagerank_max != pagerank_min:
            pagerank_normalized = {
                node: (rank - pagerank_min) / (pagerank_max - pagerank_min) 
                for node, rank in pagerank_lcc.items()
            }
        else:
            # If all PageRank values are the same, assign the midpoint value
            pagerank_normalized = {node: 0.5 for node in pagerank_lcc}

        # Define node sizes based on degree
        # Adjust the scaling factor as needed
        scaling_factor = 10  # You can adjust this value
        node_sizes = [degrees_lcc[node] * scaling_factor for node in largest_cc.nodes()]

        # Define node colors based on PageRank
        node_colors = [pagerank_normalized[node] for node in largest_cc.nodes()]

        # Choose the 'coolwarm' colormap
        cmap = plt.get_cmap('coolwarm')

        # Create the layout for the largest connected component
        pos_lcc = nx.spring_layout(largest_cc, seed=42)

        # Plot the Largest Connected Component
        plt.figure(figsize=(12, 10))
        nodes = nx.draw_networkx_nodes(
            largest_cc,
            pos_lcc,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            alpha=0.8
        )
        nx.draw_networkx_edges(largest_cc, pos_lcc, alpha=0.5)
        plt.title("Largest Connected Component: Nodes Sized by Degree and Colored by PageRank")
        plt.axis('off')
        plt.tight_layout()

        # Add a colorbar to indicate PageRank using the 'nodes' PathCollection
        cbar = plt.colorbar(nodes, shrink=0.5)
        cbar.set_label('PageRank')

        plt.show()


    # --- Additions End Here ---
if __name__ == "__main__":
    main()