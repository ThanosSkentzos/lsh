#%%
file_path="results.txt"
from lsh import get_data,jac_sim
from tqdm import tqdm
df,u,m = get_data()

#%%
values = df["movies"].values
with open(file_path,"r") as f:
    pairs = set([tuple(map(int,i.strip().split(','))) for i in f.readlines()])
pair_scores = []
for pair in pairs:
    sim = jac_sim(*values[list(pair)])
    pair_scores.append((*pair,sim))

#%%
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(sorted(pair_scores,key=lambda x:x[2]))
df[2].plot(marker='.',linestyle='none',ylabel='similarity',xlabel='pair number',title='Pair similarity')
plt.savefig("linplot.png")
#%%
y = df.index
x = sorted(df[2].values,reverse=True)
plt.figure()
plt.scatter(x,y,marker='.')
plt.yscale('log'),plt.xscale('log')
plt.xlabel("Similarity Score")
plt.ylabel("Cumulative Count")
plt.title("Cumulative Count vs Similarity Score")
plt.savefig("logplot.png")
plt.show()
# %%
from lsh import find_pairs_graph

#%%
new_pairs,new_pair_scores = find_pairs_graph(values,file_path)
df_new = pd.DataFrame(sorted(pair_scores+new_pair_scores))

# %%
