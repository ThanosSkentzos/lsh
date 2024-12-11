#%%
file_path="results.txt"
from lsh import get_data,jac_sim
from tqdm import tqdm
df,u,m = get_data()

#%%
values = df["movies"]
with open(file_path,"r") as f:
    pairs = set([tuple(map(int,i.strip().split(','))) for i in f.readlines()])
pair_scores = []
for pair in pairs:
    sim = jac_sim(*values[list(pair)])
    pair_scores.append((*pair,sim))

#%%
import pandas as pd
import matplotlib.pyplot as plt
from main import get_data
scores = pd.DataFrame(sorted(pair_scores,key=lambda x:x[2]))
pairs = scores.tail(5)[[0,1]].values
values = get_data()[0]["movies"][list(set(pairs.ravel()))]
for src,dest in pairs:
    vsrc = values[src]
    vdest = values[dest]
    inter = set(vsrc).intersection(set(vdest))
    un = set(vsrc).union(set(vdest))
    print(f"Pair ({src},{dest}): -> {len(vsrc)},{len(vdest)}",end="")
    print(f" inter/union= {len(inter)}",end="/")
    print(f"{len(un)}",end="")
    print(f" = {len(inter)/(len(un))}")
#%%
scores[2].plot(marker='.',linestyle='none',ylabel='similarity',xlabel='pair number',title='Pair similarity')
plt.savefig("images/linplot.png")
#%%
y = scores.index
x = sorted(scores[2].values,reverse=True)
plt.figure()
plt.scatter(x,y,marker='.')
plt.yscale('log'),plt.xscale('log')
plt.xlabel("Similarity Score")
plt.ylabel("Cumulative Count")
plt.title("Cumulative Count vs Similarity Score")
plt.savefig("images/logplot.png")
plt.show()
# %%
from lsh import find_pairs_graph

#%%
new_pairs,new_pair_scores = find_pairs_graph(values,file_path)
df_new = pd.DataFrame(sorted(pair_scores+new_pair_scores))

# %%
