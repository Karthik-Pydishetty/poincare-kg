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

