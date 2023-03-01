import logging
import os
import random
import torch
from torch.utils.data.dataset import Subset

import numpy as np
from tqdm import tqdm

import torch.nn as nn
from defense.weak import WeakLearner

from numba import njit, int32

# https://stackoverflow.com/questions/60894157
@njit
def mode_rand(a):
    out = np.zeros(a.shape[1], dtype=int32)
    count = np.zeros(a.shape[1], dtype=int32)
    for col in range(a.shape[1]):
        z = np.zeros(a[:,col].max()+1, dtype=int32)
        for v in a[:,col]:
            z[v]+=1
        count[col] = z.max()
        maxs = np.where(z == count[col])[0]
        out[col] = np.random.choice(maxs)
    return out, count

def filter_noise(classifier_fn, 
                 batch_size,
                 dataset, num_classes, clean_labels, 
                 optimizer_fn, scheduler_fn=None,
                 ground_truth_clean=None,
                 data_perc=.12,
                 seed=None, device=None,
                 boost=8, # alpha
                 beta=4,
                 bag=3):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    criterion = nn.CrossEntropyLoss()
    boost = max(boost, 1)
    bag = max(bag, 1)

    valloader = torch.utils.data.DataLoader(
        dataset, batch_size=1000, 
        shuffle=False, num_workers=2)

    active = dataset.targets == dataset.targets
    learner = WeakLearner(
        # problem set up
        classifier_fn=classifier_fn,
        dataset=dataset,
        active=active,
        num_classes=num_classes,
        batch_size=batch_size,

        # main hyperparams
        data_perc=1,
        refit_it=1,
        expansion=1/beta,

        # optimization
        optimizer_fn=optimizer_fn,
        criterion=criterion,
        scheduler_fn=scheduler_fn,

        # debugging only
        ground_truth_clean=ground_truth_clean,
        clean_labels=clean_labels,

        device=device
    )

    # ISPL
    learners = []
    for b in range(bag):
        active = dataset.targets == dataset.targets
        for e in range(boost):
            if sum(active) <= 2000:
                break

            dp = min(1, data_perc * len(active) / sum(active))
            wl = WeakLearner(
                # problem set up
                classifier_fn=classifier_fn,
                dataset=dataset,
                active=active,
                num_classes=num_classes,
                batch_size=batch_size,

                # main hyperparams
                data_perc=dp,
                refit_it=2 + max(1, min(3, int(1 // dp))),
                expansion=1/beta,

                # optimization
                optimizer_fn=optimizer_fn,
                criterion=criterion,
                scheduler_fn=scheduler_fn,

                # debugging only
                ground_truth_clean=ground_truth_clean,
                clean_labels=clean_labels,

                device=device
            )

            core_set, net = wl.get_core_set()
            learners.append((net, core_set))

            if boost > 1:
                active = np.logical_and(active, ~core_set)

    for net, _ in learners:
        net.eval()

    # boosting
    with torch.no_grad():
        predicted = []
        for idx, (net, core_set) in enumerate(learners):
            losses, pred = learner.test(valloader, net)
            predicted.append(pred)
        predicted = np.stack(predicted)

        vote, count = mode_rand(predicted)
        keep = np.logical_and(vote == dataset.targets, count > 1)

    # SPL
    net = None
    correct = None
    epochs = 5
    for it in range(20):
        net = learner.retrain(keep, net=net, epochs=epochs)
        with torch.no_grad():
            losses, predicted = learner.test(valloader, net)
        if correct is None:
            correct = np.ones(len(predicted)) / 2
        correct = .2 * (predicted == dataset.targets) + .8 * correct
        keep = correct > .5
    return keep, net

