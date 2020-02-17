""" Entry point.
"""
import rlog
import torch
from liftoff import parse_opts
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import src.io_utils as U
from src.data_factories import get_dsets
from src.schrodinger import SVIModel


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

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return self.out_activ(x)


class SVILoss(nn.Module):
    """ This is actually an ELBO and is of the form
        `NLL_loss + KL(q(phi | p(theta))` where `phi` are the variational
        parameters and `theta` the model's parameters.
    """

    def __init__(self, kl_div, nll_weight=None, reduction="mean"):
        super(SVILoss, self).__init__()
        self.kl_div = kl_div
        self.nll_weight = nll_weight
        self.reduction = reduction

    def forward(self, output, target):  # pylint: disable=arguments-differ
        # compute the NLL
        nll = F.nll_loss(output, target, reduction=self.reduction)
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
    total_loss, correct = 0, 0

    model.train()
    for _, (data, target) in enumerate(loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if mc_samples:
            # get the model predictions by marginalizatin
            output = model.forward(data, M=mc_samples)
        else:
            # or MLE
            output = model.forward(data)

        # compute the loss function. For SVI
        loss = criterion(output, target)

        loss.backward()

        if mc_samples is not None:
            # TODO: what was this?
            model._normals = None
        optimizer.step()

        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.item()

    total_loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    return total_loss, accuracy


def validate(loader, model, mc_samples=0):
    """ Validation routine.
    """
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
    return nll_loss, accuracy


def get_model(opt, device):
    """ Return a suitable model. """
    model = CifarConvNet(hidden_dim=512).to(device)
    if opt.mode == "SVI":
        return SVIModel(model)
    return model


def get_criterion(opt, model, nll_weight):
    """ Return a suitable loss function. """
    if opt.mode == "SVI":
        rlog.info("Loss: NLL + KL \n")
        return SVILoss(model.get_kl_div, nll_weight=nll_weight)
    rlog.info("Loss: NLL \n")
    return nn.NLLLoss()


def run(opt):
    """ Run experiment. This function is being launched by liftoff.
    """
    # logging
    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True)
    trn_log = rlog.getLogger(opt.experiment + ".train")
    val_log = rlog.getLogger(opt.experiment + ".valid")
    val_fmt = "[{:03d}][VAL]  acc={:5.2f}%  loss={:5.2f}"
    trn_fmt = "[{:03d}][TRN]  acc={:5.2f}%  loss={:5.2f}"
    # add histogram support
    if hasattr(opt, "log") and opt.log.detailed:
        trn_log.addMetrics(
            rlog.ValueMetric("std", metargs=["std"], tb_type="histogram"),
            rlog.ValueMetric("mu", metargs=["mu"], tb_type="histogram"),
        )

    # model related stuff
    device = torch.device("cuda")
    trn_set, val_set, wmp_set = get_dsets(opt)
    model = get_model(opt, device)
    optimizer = getattr(optim, opt.optim.name)(
        model.parameters(), **vars(opt.optim.args)
    )
    # batch_size
    batch_size = opt.trn_loader.batch_size

    rlog.info(U.config_to_string(opt))
    rlog.info("Model: %s", str(model))
    rlog.info("Optimizer: %s \n", str(optimizer))

    if wmp_set is not None:
        rlog.info("Warming-up on dset of size %d", len(wmp_set))
        for epoch in range(opt.warmup.epochs):
            trn_loss, trn_acc = train(
                DataLoader(wmp_set, **vars(opt.trn_loader)),
                model,
                optimizer,
                get_criterion(opt, model, len(wmp_set) // batch_size),
                mc_samples=opt.trn_mcs,
            )
            val_loss, val_acc = validate(
                DataLoader(val_set, **vars(opt.val_loader)), model, opt.tst_mcs
            )
            if isinstance(model, SVIModel):
                trn_loss_mc, trn_acc_mc = validate(
                    DataLoader(wmp_set, **vars(opt.val_loader)), model, opt.tst_mcs
                )
                # log results
                trn_log.trace(
                    step=epoch,
                    acc=trn_acc,
                    accMC=trn_acc_mc,
                    loss=trn_loss,
                    lossMC=trn_loss_mc,
                )
            else:
                trn_log.trace(
                    step=epoch,
                    acc=trn_acc,
                    loss=trn_loss,
                )

            val_log.trace(step=epoch, acc=val_acc, loss=val_loss)
            trn_log.info(trn_fmt.format(epoch, trn_acc, trn_loss))
            val_log.info(val_fmt.format(epoch, val_acc, val_loss))
            # log histogram also
            if hasattr(opt, "log") and opt.log.detailed:
                for mu, std in zip(model.mu(), model.std()):
                    trn_log.put(mu=mu, std=std)
                trn_log.trace(step=epoch, **trn_log.summarize())
                trn_log.reset()

        # maybe reset optimizer after warmup
        if opt.warmup.reset_optim:
            rlog.info("Warmup ended. Resetting optimizer.")
            optimizer = getattr(optim, opt.optim.name)(
                model.parameters(), **vars(opt.optim.args)
            )

    if wmp_set is not None:
        epochs = range(opt.warmup.epochs, opt.warmup.epochs + opt.epochs)
    else:
        epochs = range(opt.epochs)

    rlog.info("Training on dset: %s", str(trn_set))
    for epoch in epochs:
        trn_loss, trn_acc = train(
            DataLoader(trn_set, **vars(opt.trn_loader)),
            model,
            optimizer,
            get_criterion(opt, model, len(trn_set) // batch_size),
            mc_samples=opt.trn_mcs,
        )
        val_loss, val_acc = validate(
            DataLoader(val_set, **vars(opt.val_loader)), model, opt.tst_mcs
        )
        if isinstance(model, SVIModel):
            trn_loss_mc, trn_acc_mc = validate(
                DataLoader(trn_set, **vars(opt.val_loader)), model, opt.tst_mcs
            )
            # log results
            trn_log.trace(
                step=epoch,
                acc=trn_acc,
                accMC=trn_acc_mc,
                loss=trn_loss,
                lossMC=trn_loss_mc,
            )
        else:
            trn_log.trace(
                step=epoch,
                acc=trn_acc,
                loss=trn_loss,
            )

        val_log.trace(step=epoch, acc=val_acc, loss=val_loss)
        trn_log.info(trn_fmt.format(epoch, trn_acc, trn_loss))
        val_log.info(val_fmt.format(epoch, val_acc, val_loss))
        # log histogram also
        if hasattr(opt, "log") and opt.log.detailed:
            for mu, std in zip(model.mu(), model.std()):
                trn_log.put(mu=mu, std=std)
            trn_log.trace(step=epoch, **trn_log.summarize())
            trn_log.reset()


def main():
    """ Read config file using liftoff and launch experiments. You get here
        only if you launch multiple experiments using liftoff.
    """
    opt = parse_opts()
    run(opt)


if __name__ == "__main__":
    main()
