import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data.dataset import Subset

from evaluation.train import train, test
from defense.boost import filter_noise

from model.preact_resnet import PreActResNet18
from model.resnet_paper import resnet32

def try_get_list(maybe_list, idx):
    if isinstance(maybe_list, list):
        return maybe_list[idx]
    else:
        return maybe_list

def compute_poison_stats(keep, clean):
    false_pos = sum(np.logical_and(np.logical_not(keep), clean))
    false_neg = sum(np.logical_and(keep, np.logical_not(clean)))
    true_pos = sum(np.logical_and(np.logical_not(keep), np.logical_not(clean)))
    true_neg = sum(np.logical_and(keep, clean))
    return false_pos, false_neg, true_pos, true_neg

def run_defense(dataset,
        model_constructor, optimizer_constructor, 
        dataset_loader, epochs, batch_size, device, 
        scheduler_constructor=None):
    print("Dataset: " + dataset)

    clean_testset, clean_testloader = dataset_loader("clean", batch_size, train=False)
    poison_testset, poison_testloader = dataset_loader(dataset, batch_size, train=False)
    poison_trainset, poison_trainloader = dataset_loader(dataset, batch_size, train=True)
    source = poison_trainset.source
    target = poison_trainset.target

    m_ctr = try_get_list(model_constructor, 0)
    op_ctr = try_get_list(optimizer_constructor, 0)
    s_ctr = try_get_list(scheduler_constructor, 0)

    alpha = 8
    data_perc = .96 / alpha
    beta = 4
    ground_truth_clean = np.array([i in poison_trainset.clean_samples for i in range(len(poison_trainset.targets))])
    clean, net = \
            filter_noise(m_ctr,
                         batch_size,
                         poison_trainset, 10, poison_trainset.true_targets,
                         op_ctr, scheduler_fn=s_ctr,
                         data_perc=data_perc,
                         boost=alpha,
                         beta=beta,
                         ground_truth_clean=ground_truth_clean,
                         device=device)

    true_clean = np.zeros(len(poison_trainset))
    true_clean[poison_trainset.clean_samples] = 1
    false_pos, false_neg, true_pos, true_neg = compute_poison_stats(
        clean, true_clean)

    print(f"false_pos: {false_pos} | "
          f"false_neg: {false_neg} | "
          f"true_pos: {true_pos} | "
          f"true_neg: {true_neg}")

    cleanset = Subset(poison_trainset,
                      [i for i in range(len(poison_trainset)) if clean[i]])
    trainloader = torch.utils.data.DataLoader(
            cleanset, batch_size=batch_size, shuffle=True, num_workers=2)

    m_ctr = try_get_list(model_constructor, 1)
    op_ctr = try_get_list(optimizer_constructor, 1)
    s_ctr = try_get_list(scheduler_constructor, 1)

    net = m_ctr().to(device)
    optimizer = op_ctr(net.parameters())
    if s_ctr is not None:
        scheduler = s_ctr(optimizer)
    else:
        scheduler = None

    criterion = torch.nn.CrossEntropyLoss()
    train(net, criterion, optimizer, epochs, trainloader, device, 0,
          scheduler=scheduler)

    clean_accuracy, clean_misclassification = test(net, clean_testloader, device, source=source)
    print(f"clean accuracy: {clean_accuracy}")

    _, poison_misclassification = test(net, poison_testloader, device, source=source, target=target)
    p = sum(poison_misclassification) / len(poison_misclassification)
    print(f"poison misclassification: {p}")

    poison_misclassification = [p and c for p, c in zip(poison_misclassification, clean_misclassification)]
    p = sum(poison_misclassification) / len(poison_misclassification)
    print(f"targeted misclassification: {p}")


