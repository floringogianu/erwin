""" Entry point.
"""
import torch
from liftoff import parse_opts
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import src.io_utils as U
from src.data_factories import get_dsets
from src.schrodinger import SVIModel
import rlog


class CifarConvNet(nn.Module):
    """ CNN for CIFAR10 """

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.maxp1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxp2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(500, hidden_dim)
        self.out = nn.Linear(hidden_dim, 10)
        self.out_activ = nn.LogSoftmax(1)

    def forward(self, x):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return self.out_activ(x)


class SVILoss(nn.Module):
    def __init__(self, kl_div, nll_weight=None, reduction="mean"):
        """ This is actually an ELBO and is of the form
            `NLL_loss + KL(q(phi | p(theta))` where `phi` are the variational
            parameters and `theta` the model's parameters.
        """
        super(SVILoss, self).__init__()
        self.kl_div = kl_div
        self.nll_weight = nll_weight
        self.reduction = reduction

    def forward(self, input, target):
        # compute the NLL
        nll = F.nll_loss(input, target, reduction=self.reduction)
        if self.nll_weight is not None:
            nll *= self.nll_weight
        # and the kl
        kl = self.kl_div()
        return nll + kl


def train(loader, model, optimizer, criterion, mc_samples=0):
    """ Training routine.

    Args:
        loader (torch.utils.data.DataLoader): The data.
        model (schrodinger.VariationalModel): A variational wrapper over a
            user-defined MLE model.
        data_loader (torch.utils.data.DataLoader): Reads and prepares data.
        optimizer (torch.optim.Optimizer): The stochastic gradient descent
            optimization method.
        criterion: (torch.nn.Module): A loss function.
    """

    device = list(model.parameters())[0].device

    model.train()
    for _, (data, target) in enumerate(loader):

        data, target = data.to(device), target.to(device)

        if mc_samples:
            # get the model predictions by marginalizatin
            output = model.forward(data, M=mc_samples)
        else:
            # or MLE
            output = model.forward(data)

        # compute the loss function. For SVI
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()

        if mc_samples is not None:
            # TODO: what was this?
            model._normals = None
        optimizer.step()


def test(loader, model, mc_samples=0):
    device = list(model.parameters())[0].device
    nll_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            if mc_samples:
                # get the model's prediction using M samples from the posterior
                output = model.forward(data, M=mc_samples)
            else:
                output = model.forward(data)

            nll_loss += F.nll_loss(output, target, reduction="sum").item()

            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # TODO: forgot what is this about
    if mc_samples:
        model._normals = None

    nll_loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    return nll_loss, accuracy, correct


def get_model(opt, device):
    model = CifarConvNet(hidden_dim=512).to(device)
    if opt.mode == "SVI":
        return SVIModel(model)
    return model


def get_criterion(opt, model, nll_weight):
    if opt.mode == "SVI":
        return SVILoss(model.get_kl_div, nll_weight=nll_weight)
    return nn.NLLLoss()


def run(opt):
    """ Run experiment. This function is being launched by liftoff.
    """
    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True)
    # trn_log = rlog.getLogger(opt.experiment + ".train")
    val_log = rlog.getLogger(opt.experiment + ".valid")

    device = torch.device("cuda")
    trn_set, tst_set = get_dsets()
    model = get_model(opt, device)
    criterion = get_criterion(opt, model, len(trn_set) // opt.batch_size)
    optimizer = getattr(optim, opt.optim.name)(
        model.parameters(), **vars(opt.optim.args)
    )

    rlog.info(U.config_to_string(opt))
    rlog.info("Model: %s", str(model))
    rlog.info("Optimizer: %s \n", str(optimizer))

    for epoch in range(opt.epochs):
        loader = DataLoader(trn_set, batch_size=opt.batch_size, shuffle=True)
        train(loader, model, optimizer, criterion, mc_samples=opt.trn_mcs)
        tst_loss, tst_acc, tp = test(
            DataLoader(tst_set, batch_size=1024, shuffle=True),
            model,
            opt.tst_mcs,
        )
        val_log.trace(step=epoch, acc=tst_acc, loss=tst_loss)
        val_log.info(
            "[{:03d}][TEST]  acc={:5d}/{:5d} ({:5.2f}%), loss={:5.2f}".format(
                epoch, tp, len(tst_set), tst_acc, tst_loss
            )
        )


def main():
    """ Read config file using liftoff and launch experiments. You get here
        only if you launch multiple experiments using liftoff.
    """
    opt = parse_opts()
    run(opt)


if __name__ == "__main__":
    main()
