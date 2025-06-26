#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict as ddict
import pandas
import numpy as np
from numpy.random import choice
import torch as th
from torch.utils.data import Dataset as DS
from sklearn.metrics import average_precision_score
from multiprocessing.pool import ThreadPool
from functools import partial
import h5py
from tqdm import tqdm

# !pip install iopath
from iopath.common.file_io import PathManager
path_manager = PathManager()

def load_adjacency_matrix(path, format='hdf5', symmetrize=False, objects=None):
  '''
  # Loads matrix containing relations between nodes
  # Having chapter, section, subsection hierarchy here may be a good way to have that consdiered if possible

  Args:
    path: file path
    format: file format (Will need to be changed from default that is defined)

  Output:
    adajceny matrix for nodes with headers ids, neighbors, offsets, weights, objects
  '''

  # For hdf5 file format
  if format == 'hdf5':
      with path_manager.open(path, 'rb') as fin:
          with h5py.File(fin, 'r') as hf:
              return {
                  'ids': hf['ids'].value.astype('int'),
                  'neighbors': hf['neighbors'].value.astype('int'),
                  'offsets': hf['offsets'].value.astype('int'),
                  'weights': hf['weights'].value.astype('float'),
                  'objects': hf['objects'].value
                }
  # For CSV file format
  elif format == 'csv':
      df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')

      if symmetrize:
          rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
          df = pandas.concat([df, rev])

      # List/dict to remap ids/find unique ids
      idmap = {}
      idlist = []

      def convert(id):
          if id not in idmap:
              idmap[id] = len(idlist)
              idlist.append(id)
          return idmap[id]
      if objects is not None:
          objects = pandas.DataFrame.from_dict({'obj': objects, 'id': np.arange(len(objects))})
          df = df.merge(objects, left_on='id1', right_on='obj').merge(objects, left_on='id2', right_on='obj')
          df['id1'] = df['id_x']
          df['id2'] = df['id_y']
      else:
          df.loc[:, 'id1'] = df['id1'].apply(convert)
          df.loc[:, 'id2'] = df['id2'].apply(convert)
          objects = np.array(idlist)

      # Groups Edges by Node (Sort by start node then target node)
      groups = df.groupby('id1').apply(lambda x: x.sort_values(by='id2'))
      counts = df.groupby('id1').id2.size()

      ids = groups.index.levels[0].values
      offsets = counts.loc[ids].values
      offsets[1:] = np.cumsum(offsets)[:-1]
      offsets[0] = 0
      neighbors = groups['id2'].values
      weights = groups['weight'].values
      return {
          'ids' : ids.astype('int'),
          'offsets' : offsets.astype('int'),
          'neighbors': neighbors.astype('int'),
          'weights': weights.astype('float'),
          'objects': objects
        }
  else:
      raise RuntimeError(f'Unsupported file format {format}')

class Dataset(DS):

  '''
  Loads the dataset and prepares it for training/use in a KG

  Args:
    idx: Index of source and target nodes
    objects: Full list of objects (nodes)
    weights: Weight/Importnace of each source/target pait
    nnegs: Number of negative samples per positive sample
    unigram_size: Size of negative sampling table
      - should be 1e7 based on the asusmption that our data is ~100K nodes

  '''
  _neg_multiplier = 1
  _ntries = 10
  _sample_dampening = 0.75

  def __init__(self, idx, objects, weights, nnegs, unigram_size=1e8):
      assert idx.ndim == 2 and idx.shape[1] == 2
      assert weights.ndim == 1
      assert len(idx) == len(weights)
      assert nnegs >= 0
      assert unigram_size >= 0

      print('Indexing data')
      self.idx = idx
      self.nnegs = nnegs
      self.burnin = False
      self.objects = objects

      self.edge_list = [tuple(edge) for edge in idx.tolist()]

      self._weights = ddict(lambda: ddict(int))
      self._counts = np.ones(len(objects), dtype=float)
      self.max_tries = self.nnegs * self._ntries

      for i in range(idx.shape[0]):
          t, h = self.idx[i]
          self._counts[h] += weights[i]
          self._weights[t][h] += weights[i]
      self._weights = dict(self._weights)
      nents = int(np.array(list(self._weights.keys())).max())
      assert len(objects) > nents, f'Number of objects do no match'

      if unigram_size > 0:
          c = self._counts ** self._sample_dampening
          self.unigram_table = choice(
              len(objects),
              size=int(unigram_size),
              p=(c / c.sum())
          )

      self.fweights = self.weight_function

  def __len__(self):
      return self.idx.shape[0]

  def weights(self, inputs, targets):
      # no Fweights is initialized above
      return self.fweights(self, inputs, targets)

  # added function to intialize vector of ones
  @staticmethod
  def weight_function(data, input, target):
        return th.ones_like(target, dtype = th.float)

  def nnegatives(self):
      if self.burnin:
          return self._neg_multiplier * self.nnegs
      else:
          return self.nnegs

  @classmethod
  def collate(cls, batch):
      inputs, targets = zip(*batch)
      return th.cat(inputs, 0), th.cat(targets, 0)

  def edges(self):
      return self.edge_list


def eval_reconstruction(adj, model, all_vectors, k_cutoffs = [1,5,10], workers=1, progress=False):
    '''
    Reconstruction evaluation. For each object, rank its neighbors by distance,
    utlizing calls to reconstruction_worker.

    Args:
      adj (dict[int, set[int]]): Adjacency list mapping each index to its neighbors.
      model: Model with .dense_layer and .energy method.
      all_vectors (torch.Tensor[N, D]): Tensor of input vectors for all nodes.
      workers (int): number of workers to use

    Returns:
      Mean Avg Rank: Avg rank of true neighbors in sorted list (how well they are predicted)
      Mean Avg Precision: Avg precision of true neighbors in sorted list (how well they are predicted)

    '''

    # calls reconstruction worker based on the number of workers available

    with th.no_grad():
        all_vectors = model.dense_layer(all_vectors).detach()

        objects = np.array(list(adj.keys()))

        worker_fn = partial(reconstruction_worker, adj, model, all_vectors, k_cutoffs)

        if workers > 1:
            with ThreadPool(workers) as pool:

                results = pool.map(worker_fn, np.array_split(objects, workers))
                ranksum, nranks, ap_scores, iters, hits_dict = zip(*results)
                ranksum = np.sum(ranksum)
                nranks = np.sum(nranks)
                ap_scores = np.sum(ap_scores)
                iters = np.sum(iters)
                hits_k = {k: np.sum([h[k] for h in hits_dict]) / iters for k in k_cutoffs}

        else:
            ranksum, nranks, ap_scores, iters, hits_k = reconstruction_worker(
                adj, model, all_vectors, k_cutoffs, progress = progress)
            for k in k_cutoffs:
                hits_k[k] /= iters

    mean_rank = ranksum / nranks
    mean_ap   = ap_scores / iters

    return mean_rank, mean_ap, hits_k

def reconstruction_worker(adj, model, all_vectors, k_cutoffs, progress=False):

    '''
    Takes the input of a node and computes neighbors

    Args:
      adj (dict[int, set[int]]): Adjacency list mapping each index to its neighbors.
      model: Model with .dense_layer and .energy method.
      all_vectors (torch.Tensor[N, D]): Tensor of input vectors for all nodes.

    Output:
      ranksum: Number of non-neighbors ranked ahead of neighbors
      nranks: Total number of neighbors considered
      ap_scores: Total average precision scores
      iters: Number of nodes evaluated.

    Code can be modified and passed arg "objects" to evaluate only some objects
    '''

    ranksum = nranks = ap_scores = iters = 0
    num_nodes = all_vectors.size(0)
    objects = range(num_nodes)
    hits_k = {k: 0 for k in k_cutoffs}

    for idx in tqdm(objects) if progress else objects:

        neighbors = adj[idx]
        if not neighbors:
            continue

        src_vec = all_vectors[idx: idx + 1]
        all_vecs = all_vectors

        # current node distance from others, preventing self comparison
        dists = model.energy(src_vec, all_vecs).flatten()
        dists[idx] = 1e12

        sorted_idx   = dists.argsort()
        sorted_dists = dists[sorted_idx]

        ranks = np.where(np.isin(sorted_idx.cpu().numpy(), list(neighbors)))[0] + 1
        N = ranks.size

        # computes rank sum to count only non neighbors ranked higher
        ranksum += ranks.sum() - (N * (N - 1) / 2)
        nranks += N

        labels = np.isin(sorted_idx.cpu().numpy(), list(neighbors)).astype(np.int8)
        mask = np.isin(sorted_idx, neighbors)
        labels[mask] = 1

        labels = np.isin(sorted_idx.cpu().numpy(), list(neighbors)).astype(np.int8)
        scores = -sorted_dists.cpu().numpy()
        ap_scores += average_precision_score(labels, scores)

        for k in k_cutoffs:
            top_k = set(sorted_idx[:k].cpu().numpy())
            hits_k[k] += len(top_k & neighbors) / len(neighbors)


        iters += 1

    return float(ranksum), nranks, ap_scores, iters, hits_k


def eval_reconstruction_slow(adj, model, all_vectors, k_cutoffs = [1,5,10]):
    """
    Evaluates how well the model establishes the relationship between nodes (neighbors).

    Args:
      adj (dict[int, set[int]]): Adjacency list mapping each index to its neighbors.
      model: Model with .dense_layer and .energy method.
      all_vectors (torch.Tensor[N, D]): Tensor of input vectors for all nodes.

    Returns:
      Mean Avg Rank: Avg rank of true neighbors in sorted list (how well they are predicted)
      Mean Avg Precision: Avg precision of true neighbors in sorted list (how well they are predicted)
    """

    ranks = []
    ap_scores = []
    num_nodes = all_vectors.size(0)

    hits_k = {k: 0 for k in k_cutoffs}

    for i in range(num_nodes):
        neighbors = adj.get(i, set())
        if not neighbors:
            continue

        vector = all_vectors[i]
        src_vec = model.dense_layer(vector.unsqueeze(0))
        all_vecs = model.dense_layer(all_vectors)

        # Runs reconstrction to test distances and masks self comparison
        dists = model.energy(src_vec, all_vecs)
        dists[0, i] = 1e12

        # Sort closest to farthests and create labels indicating neighbor or not
        sortedidx = dists[0].argsort().cpu().numpy()
        labels = np.zeros(num_nodes)
        for n in neighbors:
            labels[n] = 1

        # Tests ranks for each neighbor
        # This can be made faster but does hurt accuracy
        masked_dists = dists.clone()
        for n in neighbors:
            masked_dists[0, n] = 1e12
        for n in neighbors:
            temp_dists = masked_dists.clone()
            temp_dists[0, n] = dists[0, n]
            rank = np.argsort(temp_dists[0].detach().cpu().numpy()).tolist().index(n) + 1
            ranks.append(rank)

        # Compute average precison socre
        ap_scores.append(average_precision_score(labels, -dists[0].detach().cpu().numpy()))

        # Compares top neighbors at each level of k
        for k in k_cutoffs:
          top_k = set(sortedidx[:k])
          hits_k[k] += len(top_k & neighbors) / len(neighbors)

    # Normalizeing hits at K
    for k in k_cutoffs:
      hits_k[k] /= num_nodes

    return np.mean(ranks), np.mean(ap_scores), hits_k


# Moved from HyperHypoEval - not currently used but could be usefull
def predict_many(model, hypo, hyper):
    '''
    Predicts hypernym hyponym pairings

    Is not actually used in KG

    Args:
      model: pass model
      hypo: list of hyponyms
      hyper: list of hypernyms

    Output:
      Distances between hypo and hyper

    '''
    device = next(model.parameters()).device

    # Converts to tensor data type if need be
    if not th.is_tensor(hypo):
        hypo_t = th.tensor(hypo, dtype = th.float32)
    if not th.is_tensor(hyper):
        hyper_t = th.tensor(hyper, dtype = th.float32)

    hypo_t = hypo_t.to(device)
    hyper_t = hyper_t.to(device)

    # Make sure shape and compairson is correct
    hypo_e = model.dense_layer(hypo_t).unsqueeze(1)   # Shapes (N, 1, p)
    hyper_e = model.dense_layer(hyper_t).unsqueeze(0) # Shapes (1, M, p)

    dists = model.energy(hypo_e, hyper_e)             # Reshaping allows for N x M result

    # Set equal elements to be infinitly far away
    if hypo_t.shape[0] == hyper_t.shape[0]:
        mask_mat = th.eye(hypo_t.size(0), dtype=th.float32, device=device)
        dists = th.where(mask_mat == 1, th.full_like(dists, 1e10), dists)

    # Might have to change to be not negative
    return -dists.detach().numpy()


# Code for Link Prediction

file_path = # File Path

adj = load_adjacency_matrix(file_path)

triples = []

# Builds out triples and assumes relation is 0
for h, neighbors in adj.items():
    for t in neighbors:
        triples.append((h, 0, t))

triples = th.tensor(triples, dtype=th.long)

n_entities = int(th.max(th.cat([triples[:, 0], triples[:, 2]])) + 1)
n_relations = int(th.max(triples[:, 1]) + 1) # Should be 1

