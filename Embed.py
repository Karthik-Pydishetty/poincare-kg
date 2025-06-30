# %%
from hype import manifolds

# %%
#model = DistanceEnergyFunction(manifolds.PoincareManifold(eps=1e-2), in_dim=384, emb_dim=10, train_size=1180)
model = DistanceEnergyFunction(manifolds.EuclideanManifold(max_norm=1.0), in_dim=384, emb_dim=30, train_size=1180)

# %%
from argparse import Namespace
#This is much easier than working with the actual argparse. When in production, change to CLI.

import torch as th
import numpy as np
import logging

from hype.checkpoint import LocalCheckpoint
from hype.rsgd import RiemannianSGD
import sys
import json
import torch.multiprocessing as mp
import shutil
from hype.hypernymy_eval import main as hype_eval

# Will said we wont need
opt = Namespace(checkpoint='mammals.pth', dset='wordnet/mammal_closure.csv',
                dim = 30, in_dim=384, manifold='poincare', model = 'distance', lr = 0.1,
                epochs = 100, batchsize = 10, negs = 50, burnin=20, dampening=0.75,
                ndproc=8,eval_each=1, fresh=True, debug=False, gpu=-1,sym=False,
                sparse=True,burnin_multiplier=0.01,neg_multiplier=1.0, quiet=False,
                lr_type='constant',train_threads=1, margin=0.1,eval='reconstruction')

# %%
opt.epoch_start = 0
import torch as th
import timeit
import gc
from tqdm import tqdm
from torch.utils import data as torch_data

_lr_multiplier = 0.1


def train(
    device,
    model,
    data,
    optimizer,
    opt,
    log,
    rank=1,
    queue=None,
    ctrl=None,
    checkpointer=None,
    progress=False,
):
    if isinstance(data, torch_data.Dataset):
        loader = torch_data.DataLoader(
            data, batch_size=opt.batchsize, shuffle=True, num_workers=opt.ndproc
        )
    else:
        loader = data

    epoch_loss = th.Tensor(len(loader))
    counts = th.zeros(model.nobjects, 1).to(device)
    for epoch in range(opt.epoch_start, opt.epochs):
        epoch_loss.fill_(0)
        data.burnin = False
        lr = opt.lr
        t_start = timeit.default_timer()
        if epoch < opt.burnin:
            data.burnin = True
            lr = opt.lr * _lr_multiplier
            '''
            if rank == 1:
                log.info(f"Burn in negs={data.nnegatives()}, lr={lr}")
            '''

        loader_iter = tqdm(loader) if progress and rank == 1 else loader
        for i_batch, (inputs, targets) in enumerate(loader_iter):
            elapsed = timeit.default_timer() - t_start
            inputs = inputs.to(device)
            targets = targets.to(device)

            # count occurrences of objects in batch
            if hasattr(opt, "asgd") and opt.asgd:
                counts = th.bincount(inputs.view(-1), minlength=model.nobjects)
                counts = counts.clamp_(min=1)
                getattr(counts, "floor_divide_", counts.div_)(inputs.size(0))
                counts = counts.double().unsqueeze(-1)

            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr, counts=counts)
            epoch_loss[i_batch] = loss.cpu().item()
        if rank == 1:
            if hasattr(data, "avg_queue_size"):
                qsize = data.avg_queue_size()
                misses = data.queue_misses()
                log.info(f"Average qsize for epoch was {qsize}, num_misses={misses}")

            if queue is not None:
                queue.put((epoch, elapsed, th.mean(epoch_loss).item(), model))
            elif ctrl is not None and epoch % opt.eval_each == (opt.eval_each - 1):
                with th.no_grad():
                    ctrl(model, epoch, elapsed, th.mean(epoch_loss).item())
            else:
                log.info(
                    "json_stats: {"
                    f'"epoch": {epoch}, '
                    f'"elapsed": {elapsed}, '
                    f'"loss": {th.mean(epoch_loss).item()}, '
                    "}"
                )
            if checkpointer and hasattr(ctrl, "checkpoint") and ctrl.checkpoint:
                checkpointer(model, epoch, epoch_loss)

        gc.collect()

# %%
device = th.device(f'cuda:{opt.gpu}' if opt.gpu >= 0 else 'cpu')
optimizer = RiemannianSGD(model.optim_params(), lr=opt.lr)

# %%
log_level = logging.DEBUG if opt.debug else logging.INFO
log = logging.getLogger('lorentz')
logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)
log.info(f'json_conf: {json.dumps(vars(opt))}')

# %%
model = model.to(device)
train(device, model, loader, optimizer, opt, log, ctrl=None, progress=not opt.quiet)

# %%
import torch as th
import numpy as np
import logging
import argparse
from hype.adjacency_matrix_dataset import AdjacencyDataset
from hype import train
#from hype.graph import load_adjacency_matrix, load_edge_list, eval_reconstruction
from hype.checkpoint import LocalCheckpoint
from hype.rsgd import RiemannianSGD
import sys
import json
import torch.multiprocessing as mp
import shutil
#from hype.graph_dataset import BatchedDataset # @manual=fbcode//deeplearning/projects/hyperbolic-embeddings:graph_dataset/hype/graph_dataset
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