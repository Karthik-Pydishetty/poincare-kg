# %%
#Because we are in base python, we do not need to worry about thread safe random number generation (had to lookup what that was. It's a c issue)
#Go right to initializing our class object

import numpy as np
import pandas as pd
import torch
import threading
import queue
from random import Random

class AdjacencyDataset:
    def __init__(self, adj, nnegs, batch_size, num_workers, burnin=False, sample_dampening=0.75):
        self.burnin = burnin
        self.nnegs = nnegs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_tries = 10*nnegs
        self.neg_multiplier = 1
        self.sample_dampening = sample_dampening
        self.queue = queue.Queue(maxsize=num_workers)

        self.ids = adj['ids']
        self.neighbors = adj['neighbors']
        self.offsets = adj['offsets']
        self.weights = adj['weights']
        self.objects = adj['objects']

        self.N = len(self.objects)
        self._setup_alias_tables() #defined below, as with all others as you will soon see

    def _setup_alias_tables(self):
        counts = np.bincount(self.neighbors, weights=self.weights, minlength=self.N)
        counts = counts ** self.sample_dampening

        prob = counts / counts.sum()
        S = prob*self.N
        A = np.arange(self.N, dtype=np.int64)

        small = [i for i in range(self.N) if S[i] < 1]
        large = [i for i in range(self.N) if S[i] > 1]

        while small and large:
            j = small.pop()
            k = large.pop()
            S[k] = S[k] - 1 + S[j]
            A[j] = k
            if S[k] < 1:
                small.append(k)
            elif S[k] > 1:
                large.append(k)

        self.S = S
        self.A = A

        def __iter__(self):
            self.perm = np.random.permutation(len(self.neighbors))
            self.current = 0
            self.join_count = 0
            self.threads = []

            for i in range(self.num_workers):
                t = threading.Thread(target=self._worker, args=(i,))
                t.start()
                self.threads.append(t)

            return self

    def _worker(self, tid):
        #Each thread has its own RNG. What we do different from cython code
        rng = Random(tid)

        while True:
            start = self.current
            self.current += self.batch_size
            if start >= len(self.neighbors):
                break

            batch = torch.LongTensor(self.batch_size, self.nnegatives()+2)
            count = self._getbatch(start, batch, rng)

            if count < self.batch_size:
                batch = batch[:count]

            self.queue.put((batch, torch.zeros(count, dtype=torch.long)))
        self.queue.put(tid)

    """Okay so here we are goig to talk about some of the main differences between the cython implementation and my implementation of the code.
    The main difference is that the cython code has a bunch of extra methods and information to track things such as diagnostics. For example, it
    will issue a warning if there are not enough threads to keep up. It basically tracks how well the the worker threads are filling the queue. In
    my code, I simply just check if all threads done and if the queue is empty. If both of thhose are true, then we stop the iteration. The performance
    metrics that the cython code gives you are pretty much irrelevant. So I  didn't decide to include them"""
    def __next__(self):
        if self.join_count == len(self.threads) and self.queue.empty():
            raise StopIteration

        item = self.queue.get()
        if isinstance(item, int):

            tid = item
            self.join_count += 1
            self.threads[tid].join()
            return self.__next__()

        return item

    def __len__(self):
        return int(np.ceil(len(self.neighbors)/self.batch_size))

    def random_node(self, rng):
        if self.burnin:
            u = rng.random()*self.N
            i = int(u)
            if (u - i) > self.S[i]:
                return int(self.A[i])
            else:
                return i
        else:
            return rng.randrange(self.N)

    def binary_search(self, target, arr, left, right, approx=False):
        l, r = left, right
        while l <= r:
            mid = (l + r) // 2
            if not approx and arr[mid] == target:
                return mid
            if approx:
                if arr[mid] <= target and (mid == r or arr[mid+1] > target):
                    return mid
            if arr[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return (0 if approx else -1)

    def _getbatch(self, idx, batch, rng):
        i = 0
        M = len(self.neighbors)

        while idx < M and i < self.batch_size:
            neighbor_idx = self.perm[idx]
            nodeidx = self.binary_search(neighbor_idx, self.offsets, 0, self.N-1, True)

            t = int(self.ids[nodeidx])
            h = int(self.neighbors[neighbor_idx])

            left = int(self.offsets[nodeidx])
            right = int(self.offsets[nodeidx + 1] - 1) if nodeidx + 1 < self.N else M - 1

            batch[i, 0] = t
            batch[i, 1] = h

            negs = set()
            col = 2
            tries = 0
            while tries < self.max_tries and col < self.nnegatives() + 2:
                rn = self.random_node(rng)
                rn_idx = self.binary_search(rn, self.neighbors, left, right, False)
                if rn != t and (rn_idx == -1 or self.weights[rn_idx] < self.weights[neighbor_idx]):
                    if rn not in negs:
                        batch[i, col] = rn
                        negs.add(rn)
                        col += 1
                tries += 1

            if col == 2:
                batch[i, col] = t
                col += 1

            while col < self.nnegatives() + 2:
                idx_choice = rng.randrange(col-2)
                batch[i, col] = batch[i, 2 + idx_choice]
                col += 1

            idx += 1
            i += 1

        return i

    def nnegatives(self):
        if self.burnin:
            return int(self.neg_multiplier * self.nnegs)
        return self.nnegs



