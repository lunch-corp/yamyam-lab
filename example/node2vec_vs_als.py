import implicit
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from scipy.sparse import csr_matrix

num_users = 10
num_items = 10

# make user x item binary matrix
data = []
for i in range(num_items):
    likes = [0] * num_items
    likes[i] += 1
    if i != 0:
        likes[i - 1] += 1
    data.append(likes)
user_item = csr_matrix(data)

print("user x item matrix")
print(user_item.toarray())

# create networkx graph
G = nx.Graph()

# add node
for i in range(num_users):
    G.add_node(
        f"user_{i}",
    )
    G.add_node(f"item_{i}")

# add edge
for u in range(len(user_item.indptr) - 1):
    for i in user_item.indices[user_item.indptr[u] : user_item.indptr[u + 1]]:
        G.add_edge(
            f"user_{u}",
            f"item_{i}",
        )

plt.figure(figsize=(20, 10))
nx.draw_networkx(G)
plt.show()

# train node embedding!
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# get closest nodes with user_5
for res in model.wv.most_similar("user_5", topn=20):
    print(res)
"""
[('item_4', 0.8717125654220581),
 ('item_5', 0.8662003874778748),
 ('user_6', 0.8148666024208069),
 ('user_4', 0.7902359962463379),
 ('item_3', 0.6979408860206604),
 ('item_6', 0.6480956077575684),
 ('user_3', 0.5772055387496948),
 ('user_7', 0.5155405402183533),
 ('item_7', 0.43306705355644226),
 ('item_2', 0.4155113399028778),
 ('user_8', 0.34484121203422546),
 ('user_2', 0.34050002694129944),
 ('item_8', 0.2771895229816437),
 ('item_1', 0.2479073405265808),
 ('item_9', 0.2155066281557083),
 ('user_9', 0.21406789124011993),
 ('user_1', 0.1909690946340561),
 ('user_0', 0.10896944254636765),
 ('item_0', 0.08388140797615051)]
"""

# train als!
params = {"factors": 64, "iterations": 100, "regularization": 0.01, "random_state": 42}
als = implicit.als.AlternatingLeastSquares(**params)
als.fit(user_item)

indices, distance = als.recommend(np.arange(10), user_item)
# top reocommended item index for user_5
print(indices[5])
"""
array([6, 3, 8, 1, 0, 2, 9, 7, 5, 4], dtype=int32)
"""
