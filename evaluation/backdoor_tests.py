import sys
import torch
from evaluation.run_defense import run_defense
from data.cifar10 import cifar10_loader
from model.preact_resnet import PreActResNet18
from model.resnet_paper import resnet32
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

pairs = [(0, 2), (1, 3), (3, 5), (7, 4), (2, 5), (8, 6), (9, 2), (3, 7)]
poison_levels = [0.05, 0.1, 0.2]
def get_datasets():
    ds = []
    for poison in poison_levels:
        for source, target in pairs:
            ds.append(f"datasets/cifar-backdoor-{source}-to-{target}-{poison}.pickle")
    return ds
    
def run(dataset, model):
    if "32" in model:
        model_ctor = resnet32
    elif "18" in model:
        model_ctor = PreActResNet18
    else:
        assert False, f"model not parseable from {model}"

    if model_ctor == resnet32:
        train_op_ctr = lambda parameters: torch.optim.SGD(
                parameters, lr=0.1, momentum=0.9, weight_decay=1e-4)
        train_s_ctr = lambda optimizer, e=200: torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[100, 150], gamma=0.1)
        filter_op_ctr = train_op_ctr
        filter_s_ctr = lambda optimizer, e=100: torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[75, 90], gamma=0.1)
    elif model_ctor == PreActResNet18:
        train_op_ctr = lambda parameters: torch.optim.SGD(
                parameters, lr=0.02, momentum=0.9, weight_decay=5e-4)
        train_s_ctr = lambda optimizer, e=200: torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[100, 150, 180], gamma=0.1)
        filter_op_ctr = train_op_ctr
        filter_s_ctr = train_s_ctr

    testsets = ['clean', dataset]
    run_defense(
        dataset,
        model_ctor, 
        [filter_op_ctr, train_op_ctr], cifar10_loader, 
        200, 128, 
        device, [filter_s_ctr, train_s_ctr])

if __name__ == "__main__":
    exp_id = int(sys.argv[1])
    model = sys.argv[2]

    dataset = get_datasets()[exp_id]
    run(dataset, model)

