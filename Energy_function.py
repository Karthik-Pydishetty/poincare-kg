# %%
class EnergyFunction(torch.nn.Module):  # compute distance or score between vectors using curved space
    def __init__(self, manifold, in_dim, emb_dem, size=None, sparse=False, **kwargs):
        super().__init__()
        self.manifold = manifold
        self.embedding_dim = emb_dim

        self.dense_layer = torch.nn.Linear(in_dim, emb_dim)  # initialized in forward based on input shape

        # self.lt = self.manifold.allocate_lt(size, dim, sparse)  # make a table of vectors, one vector for each object
        # self.nobjects = size
        # self.manifold.init_weights(self.lt)  # fills vector table with random starting points

    def forward(self, inputs):  # get called vectors and compute how related they are

        e = self.dense_layer(inputs)  # obtain vectors
        with torch.no_grad():
            e = self.manifold.normalize(e)  # fix vectors to stay inside ball
        o = e.narrow(1, 1, e.size(1) - 1)  # get every vector but the first
        s = e.narrow(1, 0, 1).expand_as(o)  # take subject vector and copy so it lines with each object to compare
        return self.energy(s, o).squeeze(-1)

    def optim_params(self):
        return [
            {
                "params": self.dense_layer.parameters(),
                # Had self.lt.parameters(), changed to dense_layer
                "rgrad": self.manifold.rgrad,
                "expm": self.manifold.expm,  # when vector is changed follow curved space
                "logm": self.manifold.logm,  # turn point from curved space into reg vector
                "ptransp": self.manifold.ptransp,  # move vector from one point to another without twisting or stretching
            }
        ]

    def loss_function(self, inp, target, **kwargs):
        raise NotImplementedError


# %%

class DistanceEnergyFunction(EnergyFunction): #uses distance between vectors to measure how related they are

  def energy(self, subject_vectors, object_vectors): #measure curved space distance between vectors
    return self.manifold.distance(subject_vectors, object_vectors)

  def loss(self, input_scores, target_labels, **kwargs):
    return F.cross_entropy(input_scores.neg(), target_labels)
  
# %%

# Not used - but keep incase

class EntailmentConeEnergyFunction(EnergyFunction):
    def __init__(self, *args, margin=0.1, **kwargs):  # Set up with an optional margin hyperparameter
      super().__init__(*args, **kwargs)  # Initialize parent EnergyFunction
      assert self.manifold.K is not None, "K cannot be none for EntailmentConeEnergyFunction"  # Ensure curvature K is defined
      assert hasattr(self.manifold, "angle_at_u"), "Missing `angle_at_u` method"  # Require angle calculation method on manifold
      self.margin = margin  # Store margin used in custom loss

    def energy(self, subject_vectors, object_vectors):  # Compute angle-based energy
      energy = self.manifold.angle_at_u(object_vectors, subject_vectors) - self.manifold.half_aperture(object_vectors)  # How far outside the cone the subject lies
      return energy.clamp(min=0)  # Clamp to zero to ignore points inside the cone

    def loss(self, input_scores, target_labels, **kwargs):  # Custom loss for entailment using margin-based ranking
      loss = input_scores[:, 0].clamp_(min=0).sum()  # Penalize positive samples that fall outside the cone
      loss += (self.margin - input_scores[:, 1:]).clamp_(min=0).sum()  # Penalize negatives that fall too close to the cone
      return loss / input_scores.numel()  # Normalize by total number of scores


# %%

#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import numpy as np
import logging
import argparse
from hype.adjacency_matrix_dataset import AdjacencyDataset
from hype import train
from hype.graph import load_adjacency_matrix, load_edge_list, eval_reconstruction
from hype.checkpoint import LocalCheckpoint
from hype.rsgd import RiemannianSGD
import sys
import json
import torch.multiprocessing as mp
import shutil
from hype.graph_dataset import BatchedDataset # @manual=fbcode//deeplearning/projects/hyperbolic-embeddings:graph_dataset/hype/graph_dataset
from hype import MANIFOLDS, MODELS, build_model
from hype.hypernymy_eval import main as hype_eval

th.manual_seed(42)
np.random.seed(42)


def reconstruction_eval(adj, opt, epoch, elapsed, loss, pth, best):
    chkpnt = th.load(pth, map_location='cpu')
    model = build_model(opt, chkpnt['embeddings'].size(0))
    model.load_state_dict(chkpnt['model'])

    meanrank, maprank = eval_reconstruction(adj, model)
    sqnorms = model.manifold.norm(model.lt)
    return {
        'epoch': epoch,
        'elapsed': elapsed,
        'loss': loss,
        'sqnorm_min': sqnorms.min().item(),
        'sqnorm_avg': sqnorms.mean().item(),
        'sqnorm_max': sqnorms.max().item(),
        'mean_rank': meanrank,
        'map_rank': maprank,
        'best': bool(best is None or loss < best['loss']),
    }


def hypernymy_eval(epoch, elapsed, loss, pth, best):
    _, summary = hype_eval(pth, cpu=True)
    return {
        'epoch': epoch,
        'elapsed': elapsed,
        'loss': loss,
        'best': bool(
            best is None or summary['eval_hypernymy_avg'] > best['eval_hypernymy_avg'])
        ,
        **summary
    }


def async_eval(adj, q, logQ, opt):
    best = None
    while True:
        temp = q.get()
        if temp is None:
            return

        if not q.empty():
            continue

        epoch, elapsed, loss, pth = temp
        if opt.eval == 'reconstruction':
            lmsg = reconstruction_eval(adj, opt, epoch, elapsed, loss, pth, best)
        elif opt.eval == 'hypernymy':
            lmsg = hypernymy_eval(epoch, elapsed, loss, pth, best)
        else:
            raise ValueError(f'Unrecognized evaluation: {opt.eval}')
        best = lmsg if lmsg['best'] else best
        logQ.put((lmsg, pth))


# Adapated from:
# https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
class Unsettable(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(Unsettable, self).__init__(option_strings, dest, nargs='?', **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        val = None if option_string.startswith('-no') else values
        setattr(namespace, self.dest, val)


def main():
    parser = argparse.ArgumentParser(description='Train Hyperbolic Embeddings')
    parser.add_argument('-checkpoint', default='/tmp/hype_embeddings.pth',
                        help='Where to store the model checkpoint')
    parser.add_argument('-dset', type=str, required=True,
                        help='Dataset identifier')
    parser.add_argument('-dim', type=int, default=20,
                        help='Embedding dimension')
    parser.add_argument('-manifold', type=str, default='lorentz',
                        choices=MANIFOLDS.keys())
    parser.add_argument('-model', type=str, default='distance',
                        choices=MODELS.keys(), help='Energy function model')
    parser.add_argument('-lr', type=float, default=1000,
                        help='Learning rate')
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('-batchsize', type=int, default=12800,
                        help='Batchsize')
    parser.add_argument('-negs', type=int, default=50,
                        help='Number of negatives')
    parser.add_argument('-burnin', type=int, default=20,
                        help='Epochs of burn in')
    parser.add_argument('-dampening', type=float, default=0.75,
                        help='Sample dampening during burnin')
    parser.add_argument('-ndproc', type=int, default=8,
                        help='Number of data loading processes')
    parser.add_argument('-eval_each', type=int, default=1,
                        help='Run evaluation every n-th epoch')
    parser.add_argument('-fresh', action='store_true', default=False,
                        help='Override checkpoint')
    parser.add_argument('-debug', action='store_true', default=False,
                        help='Print debuggin output')
    parser.add_argument('-gpu', default=0, type=int,
                        help='Which GPU to run on (-1 for no gpu)')
    parser.add_argument('-sym', action='store_true', default=False,
                        help='Symmetrize dataset')
    parser.add_argument('-maxnorm', '-no-maxnorm', default='500000',
                        action=Unsettable, type=int)
    parser.add_argument('-sparse', default=False, action='store_true',
                        help='Use sparse gradients for embedding table')
    parser.add_argument('-burnin_multiplier', default=0.01, type=float)
    parser.add_argument('-neg_multiplier', default=1.0, type=float)
    parser.add_argument('-quiet', action='store_true', default=False)
    parser.add_argument('-lr_type', choices=['scale', 'constant'], default='constant')
    parser.add_argument('-train_threads', type=int, default=1,
                        help='Number of threads to use in training')
    parser.add_argument('-margin', type=float, default=0.1, help='Hinge margin')
    parser.add_argument('-eval', choices=['reconstruction', 'hypernymy'],
                        default='reconstruction', help='Which type of eval to perform')
    opt = parser.parse_args()

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger('lorentz')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    if opt.gpu >= 0 and opt.train_threads > 1:
        opt.gpu = -1
        log.warning(f'Specified hogwild training with GPU, defaulting to CPU...')

    # set default tensor type
    th.set_default_tensor_type('torch.DoubleTensor')
    # set device
    device = th.device(f'cuda:{opt.gpu}' if opt.gpu >= 0 else 'cpu')

    if 'csv' in opt.dset:
        log.info('Using edge list dataloader')
        idx, objects, weights = load_edge_list(opt.dset, opt.sym)
        data = BatchedDataset(idx, objects, weights, opt.negs, opt.batchsize,
            opt.ndproc, opt.burnin > 0, opt.dampening)
    else:
        log.info('Using adjacency matrix dataloader')
        dset = load_adjacency_matrix(opt.dset, 'hdf5')
        log.info('Setting up dataset...')
        data = AdjacencyDataset(dset, opt.negs, opt.batchsize, opt.ndproc,
            opt.burnin > 0, sample_dampening=opt.dampening)
        objects = dset['objects']

    model = build_model(opt, len(objects))

    # set burnin parameters
    data.neg_multiplier = opt.neg_multiplier
    train._lr_multiplier = opt.burnin_multiplier

    # Build config string for log
    log.info(f'json_conf: {json.dumps(vars(opt))}')

    if opt.lr_type == 'scale':
        opt.lr = opt.lr * opt.batchsize

    # setup optimizer
    optimizer = RiemannianSGD(model.optim_params(), lr=opt.lr)

    # setup checkpoint
    checkpoint = LocalCheckpoint(
        opt.checkpoint,
        include_in_all={'conf' : vars(opt), 'objects' : objects},
        start_fresh=opt.fresh
    )

    # get state from checkpoint
    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    opt.epoch_start = state['epoch']

    adj = {}
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}

    controlQ, logQ = mp.Queue(), mp.Queue()
    control_thread = mp.Process(target=async_eval, args=(adj, controlQ, logQ, opt))
    control_thread.start()

    # control closure
    def control(model, epoch, elapsed, loss):
        """
        Control thread to evaluate embedding
        """
        lt = model.w_avg if hasattr(model, 'w_avg') else model.lt.weight.data
        model.manifold.normalize(lt)

        checkpoint.path = f'{opt.checkpoint}.{epoch}'
        checkpoint.save({
            'model': model.state_dict(),
            'embeddings': lt,
            'epoch': epoch,
            'model_type': opt.model,
        })

        controlQ.put((epoch, elapsed, loss, checkpoint.path))

        while not logQ.empty():
            lmsg, pth = logQ.get()
            shutil.move(pth, opt.checkpoint)
            if lmsg['best']:
                shutil.copy(opt.checkpoint, opt.checkpoint + '.best')
            log.info(f'json_stats: {json.dumps(lmsg)}')

    control.checkpoint = True
    model = model.to(device)
    if hasattr(model, 'w_avg'):
        model.w_avg = model.w_avg.to(device)
    if opt.train_threads > 1:
        threads = []
        model = model.share_memory()
        args = (device, model, data, optimizer, opt, log)
        kwargs = {'ctrl': control, 'progress' : not opt.quiet}
        for i in range(opt.train_threads):
            kwargs['rank'] = i
            threads.append(mp.Process(target=train.train, args=args, kwargs=kwargs))
            threads[-1].start()
        [t.join() for t in threads]
    else:
        train.train(device, model, data, optimizer, opt, log, ctrl=control,
            progress=not opt.quiet)
    controlQ.put(None)
    control_thread.join()
    while not logQ.empty():
        lmsg, pth = logQ.get()
        shutil.move(pth, opt.checkpoint)
        log.info(f'json_stats: {json.dumps(lmsg)}')


if __name__ == '__main__':
    main()
