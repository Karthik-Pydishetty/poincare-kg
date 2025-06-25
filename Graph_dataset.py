import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

%%bash
mkdir -p hype
cat > hype/graph_dataset_vectors.py << 'PYCODE'
import torch
import numpy as np
from torch.utils.data import IterableDataset

class VectorBatchedDataset(IterableDataset):
    """
    Yields batches of precomputed node embeddings (positive + negatives)
    for contrastive training.
    """
    def __init__(self, idx, embeddings, weights, nnegs, batch_size,
                 burnin=False, sample_dampening=0.75):
        assert idx.ndim == 2 and idx.shape[1] == 2
        assert embeddings.ndim == 2
        assert weights.ndim == 1 and weights.shape[0] == idx.shape[0]
        self.idx = idx.astype(np.int64)
        self.emb = embeddings.astype(np.float32)
        self.weights = weights.astype(np.float64)
        self.nnegs = int(nnegs)
        self.batch_size = int(batch_size)
        self.burnin = bool(burnin)
        self.sample_dampening = float(sample_dampening)
        self.N, self.D = self.emb.shape
        self._build_weights()
        self.max_tries = 10 * self.nnegs

    def _build_weights(self):
        self.counts = np.zeros(self.N, dtype=np.float64)
        self._weights = [dict() for _ in range(self.N)]
        for (t,h),w in zip(self.idx, self.weights):
            self.counts[h] += w
            self._weights[int(t)][int(h)] = float(w)
        self.counts = np.power(self.counts, self.sample_dampening)
        if self.burnin:
            prob = (self.counts/np.sum(self.counts))*self.N
            alias = np.arange(self.N, dtype=np.int64)
            small = [i for i,p in enumerate(prob) if p<1.0]
            large = [i for i,p in enumerate(prob) if p>1.0]
            while small and large:
                s = small.pop(); l = large.pop()
                alias[s] = l
                prob[l] = prob[l] - (1.0 - prob[s])
                if prob[l] < 1.0: small.append(l)
                elif prob[l] > 1.0: large.append(l)
            self.alias_prob, self.alias_alias = prob, alias

    def _alias_sample(self):
        i = np.random.randint(self.N)
        return int(self.alias_alias[i]) if np.random.rand()>self.alias_prob[i] else i

    def __iter__(self):
        perm = np.random.permutation(len(self.idx))
        for start in range(0, len(perm), self.batch_size):
            batch_idx = perm[start:start+self.batch_size]
            bsize = len(batch_idx)
            batch = np.empty((bsize, 2 + self.nnegs, self.D), dtype=np.float32)
            for j,ei in enumerate(batch_idx):
                t,h = self.idx[ei]
                batch[j,0], batch[j,1] = self.emb[t], self.emb[h]
                negs,tries = set(),0
                wt = self._weights[t].get(h,0.0)
                while tries < self.max_tries and len(negs) < self.nnegs:
                    n = self._alias_sample() if self.burnin else np.random.randint(self.N)
                    if n!=t and ((n not in self._weights[t]) or self._weights[t][n]<wt):
                        negs.add(n)
                    tries+=1
                if not negs: negs.add(t)
                negs = list(negs)
                for k in range(self.nnegs):
                    idx_s = negs[k] if k<len(negs) else negs[np.random.randint(len(negs))]
                    batch[j,2+k] = self.emb[idx_s]
            yield torch.from_numpy(batch), torch.zeros(bsize, dtype=torch.long)

    def __len__(self):
        return int(np.ceil(len(self.idx)/float(self.batch_size)))
PYCODE


# new cell

# --- Cell 4: Load mammal embeddings & filtered edges, build emb_vectors + idx_array + weights ---

import pandas as pd
import numpy as np

# 1) Load the unique set of mammal noun embeddings
#    (this comes from your df_mammals screenshot: 1180 rows × 2 columns)
df_mammals = pd.read_csv(
    "/content/poincare-embeddings/wordnet/mammals_embedding.csv",
    dtype={"noun_id": str, "embedding": str},
    low_memory=False
)

# Parse each embedding-string into a float numpy array
def parse_vec(s):
  #This was wrong, no idea why
  return np.fromstring(s.strip("[]"), sep=" ", dtype=np.float32)
  #return np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)

df_mammals["emb_vec"] = df_mammals["embedding"].fillna("[]").apply(parse_vec)

# Stack into an [N, D] array
emb_list = df_mammals["emb_vec"].to_list()
D = emb_list[0].size
emb_vectors = np.vstack([
    v if v.size == D else np.zeros(D, dtype=np.float32)
    for v in emb_list
]).astype(np.float32)

# Build a mapping from noun_id → row index
nodes       = df_mammals["noun_id"].tolist()   # list of unique synset strings
noun2row    = {nid:i for i,nid in enumerate(nodes)}

print(f"Loaded {len(nodes)} unique mammal nodes, embedding dim = {D}")

# 2) Load the filtered edges (many repeats of noun IDs)
df_filtered = pd.read_csv(
    "/content/poincare-embeddings/wordnet/mammal_closure.csv",
    dtype={"id1": str, "id2": str, "weight": float}
)

# Convert each string‐pair into integer indices into emb_vectors
idx_array = np.array([
    [noun2row[src], noun2row[tgt]]
    for src,tgt in zip(df_filtered["id1"], df_filtered["id2"])
], dtype=np.int64)

print(f"Loaded {len(idx_array)} edges (with repeats)")

# 3) Build the weights array
weights = df_filtered["weight"].to_numpy(dtype=np.float64)

# now idx_array, emb_vectors, weights are ready for VectorBatchedDataset
