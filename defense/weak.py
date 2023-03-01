import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Subset
from torch.cuda.amp import autocast, GradScaler

class WeakLearner(object):
    """
    Weak Learner with ISPL objective 
    """

    def __init__(self, classifier_fn, dataset, active,
                 num_classes, batch_size,
                 data_perc,
                 refit_it,
                 optimizer_fn, criterion, expansion=.25,
                 scheduler_fn=None,
                 ground_truth_clean=None, clean_labels=None, 
                 device=None, amp=True):
        self.classifier_fn = classifier_fn

        self.dataset = dataset
        self.active = active
        self.num_classes = num_classes
        self.dataset_size = len(dataset)

        self.batch_size = batch_size
        self.data_perc = data_perc
        self.refit_it = refit_it

        self.optimizer_fn = optimizer_fn
        self.criterion = criterion
        self.scheduler_fn = scheduler_fn

        self.ground_truth_clean = ground_truth_clean
        self.clean_labels = clean_labels
        self.amp = amp

        self.warm_up = 8
        self.retrain_epochs = 40

        self.loss_m = None
        self.decay = .9
        self.expansion = expansion

        self.device = device if device is not None else torch.device("cpu")
    
    def get_core_set(self):
        loss_fn = nn.CrossEntropyLoss(reduction="none")

        active_labels = self.dataset.targets[self.active]
        active_clean = self.ground_truth_clean[self.active]
        indices_to_keep = np.copy(self.active)

        if len(self.active) == sum(self.active):
            indices_to_keep[self.active] = np.random.binomial(n=1, p=0.5, size=sum(self.active))

        filtered = Subset(self.dataset, 
            [i for i in range(self.dataset_size) if self.active[i]])

        valloader = torch.utils.data.DataLoader(
            filtered, batch_size=self.batch_size * 10, 
            shuffle=False, num_workers=2)

        if self.data_perc >= .999:
            self.refit_it = 0
            active_keep = self.active[self.active]

        for it in range(self.refit_it):
            print(f"refit it {it}")

            if it == 0:
                net = self.classifier_fn().to(self.device)
                if self.amp:
                    scaler = GradScaler()
                optimizer = \
                    self.optimizer_fn(net.parameters())
                if self.scheduler_fn is not None:
                    scheduler = self.scheduler_fn(optimizer, 2 * self.refit_it)
                else:
                    scheduler = None
                epochs = self.warm_up
            else:
                epochs = 4

            filtered = Subset(self.dataset, 
                [i for i in range(self.dataset_size) if self.active[i] and indices_to_keep[i]])

            trainloader = torch.utils.data.DataLoader(
                filtered, batch_size=self.batch_size, 
                shuffle=True, num_workers=2, drop_last=True)

            self.train(trainloader, net, optimizer, scaler, scheduler, epochs)
            losses, predicted = self.test(valloader, net, loss_fn)

            active_keep = \
                self.get_trimmed_set(
                    it,
                    losses, 
                    predicted, 
                    indices_to_keep[self.active],
                    active_labels)
            indices_to_keep = np.copy(self.active)
            active_train = np.copy(active_keep)

            p = self.expansion 
            if p < 1:
                discard = np.random.binomial(n=1, size=len(active_keep), p=1-p)
                active_train[discard == 1] = 0
            indices_to_keep[self.active] = active_train

            train_keep = indices_to_keep[self.active]
            poison_class = []
            clean_class = []
            poison_class_perc = []
            for i in range(self.num_classes):
                mask = active_labels == i
                _poison_class = sum(
                    [i and not j \
                     for i, j \
                     in zip(train_keep[mask], 
                            active_clean[mask])])
                _clean_class = sum(
                    [i and j \
                     for i, j \
                     in zip(train_keep[mask], 
                            active_clean[mask])])
                poison_class.append(_poison_class)
                clean_class.append(_clean_class)
            print(f"poison_class: {poison_class}")
            print(f"clean_class: {clean_class}")

        indices_to_keep = np.copy(self.active)
        indices_to_keep[self.active] = active_keep
        net = self.retrain(indices_to_keep, epochs=self.retrain_epochs)

        poison_class = []
        clean_class = []
        poison_class_perc = []
        for i in range(self.num_classes):
            mask = active_labels == i
            _poison_class = sum(
                [i and not j \
                 for i, j \
                 in zip(active_keep[mask], 
                        active_clean[mask])])
            _clean_class = sum(
                [i and j \
                 for i, j \
                 in zip(active_keep[mask], 
                        active_clean[mask])])
            poison_class.append(_poison_class)
            clean_class.append(_clean_class)

        print(f"final poison_class: {poison_class}")
        print(f"final clean_class: {clean_class}")

        return indices_to_keep, net

    def retrain(self, indices_to_keep, net=None, epochs=None):
        if epochs is None:
            epochs = self.refit_it * 2
        if net is None:
            net = self.classifier_fn().to(self.device)
        if self.amp:
            scaler = GradScaler()
        optimizer = \
            self.optimizer_fn(net.parameters())
        if self.scheduler_fn is not None:
            scheduler = self.scheduler_fn(optimizer, epochs)
        else:
            scheduler = None
        filtered = Subset(self.dataset, 
            [i for i in range(self.dataset_size) if self.active[i] and indices_to_keep[i]])

        trainloader = torch.utils.data.DataLoader(
            filtered, batch_size=self.batch_size, 
            shuffle=True, num_workers=2, drop_last=True)

        self.train(trainloader, net, optimizer, scaler, scheduler, epochs)
        return net

    def get_trimmed_set(self, it,
            losses, predicted, indices_to_keep, labels):
        losses = np.nan_to_num(losses, nan=np.nanmean(losses))

        if self.loss_m is None:
            loss_m = np.zeros(len(losses))
            self.loss_m = loss_m
        else:
            loss_m = self.loss_m

        losses = losses * (1 - self.decay) + loss_m * self.decay
        self.loss_m = np.copy(losses)

        # 1 to 0
        alpha = max(min((self.refit_it - 2 - it) / (self.refit_it - 2), 1), 0)

        thresh = min(3, max(1, self.refit_it - 2 - it)) * self.data_perc
        thresh = max(0, min(1, thresh))

        class_thresh = thresh / 8

        keep = losses == losses
        for i in range(self.num_classes):
            mask = self.dataset.targets[self.active] == i
            if sum(mask) == 0:
                continue
            cutoff = np.quantile(losses[mask], [0, class_thresh, 1])[1]
            keep[mask] = losses[mask] <= cutoff

        cutoff = np.quantile(losses, [0, thresh, 1])[1]
        keep = np.logical_or(keep, losses <= cutoff)

        return keep

    def train(self, trainloader, net, optimizer, scaler, scheduler, num_epochs=10):
        net.train()
        num_batches = len(trainloader)
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = \
                    inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                if self.amp:
                    with autocast():
                        outputs = net(inputs)
                        loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = net(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            if scheduler is not None:
                scheduler.step()

        return net

    def test(self, valloader, net, loss_fn=None):
        if loss_fn is None: loss_fn = nn.CrossEntropyLoss(reduction="none")

        losses = []
        predicted = []

        net.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valloader):
                inputs, labels = \
                    inputs.to(self.device), labels.to(self.device)

                outputs = net(inputs)
                _, pred = torch.max(outputs.data, 1)
                predicted.append(pred)

                loss = loss_fn(outputs, labels)
                losses.append(loss)

        losses = torch.cat(losses).cpu().numpy()
        predicted = torch.cat(predicted).cpu().numpy()
        return losses, predicted
