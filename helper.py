import time
import pickle
import torch
from torchvision import datasets, transforms


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def clear_cache():
    with torch.no_grad():
        torch.cuda.empty_cache()


def load_cifar10_data(batch=60):
    """
    Load cifar10 data
    :param batch:
    :return: loaders for train, validation and test
    """
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_dataset = datasets.CIFAR10('data',
                                     download=True,
                                     train=True,
                                     transform=train_transform, )
    test_dataset = datasets.CIFAR10(
        root='data', train=False,
        download=True, transform=test_transform,
    )
    train_set, val_set = torch.utils.data.random_split(cifar_dataset, [45000, 5000])
    loader_train = torch.utils.data.DataLoader(train_set, batch_size=batch,
                                               shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch,
                                             shuffle=False, num_workers=2)

    loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch,
                                              shuffle=False, num_workers=2)
    return loader_train, val_loader, loader_test


def train(model, train_loader, criterion, optimizer, epoch_log, device='cpu', log_msg=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    train_iter = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        prec1, prec5 = accuracy(logits.data, labels, topk=(1, 5))

        losses.update(loss.data, images.size(0))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if log_msg:
            print(f'{epoch_log} \n'
                  f'Iter: [{i}/{train_iter}] \n'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \n'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f}) \n'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f}) \n'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \n'
                  f'Prec@5 {top5.val:.3f} ({top5.avg:.3f}) \n')


def valid(model, valid_loader, criterion, device='cpu', log_msg=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()
    valid_iter = len(valid_loader)

    for i, (images, labels) in enumerate(valid_loader):

        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))

        losses.update(loss.data, images.size(0))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if log_msg:
            print(f'Iter: [{i}/{valid_iter}]\n'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\n'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\n'
                  f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n')

    return top1.avg, top5.avg


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
