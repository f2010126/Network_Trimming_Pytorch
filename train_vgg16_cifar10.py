from torchvision import datasets, transforms
import time
import torch
import argparse
import torch.nn as nn
from vgg import vgg16
from helper import valid, train
from torchsummary import summary


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


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser(description='Full training the VGG16 for Cifar10')
    parser.add_argument('--data_path', type=str, default='./',
                        help='Path to root dataset folder ')
    parser.add_argument('--save_model', type=str, default='vgg16_full_train_model',
                        help='Path to model save')

    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=1)
    args = parser.parse_args()

    # gives the loaders
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    # of training hyperparameters in experiments: {base-lr: 0.001, gamma: 0.1, step-size: 3000}
    print(f"Training VGG16 with {args}")
    train_loader, valid_loader, test_loader = load_cifar10_data()

    model = vgg16(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                weight_decay=1e-4)

    # show_summary(model) # what is this one?
    summary(model, (3, 224, 224))

    best_top1 = 0

    for e in range(args.epoch):
        train(model,
              train_loader,
              criterion,
              optimizer,
              f"EPOCH : [{e + 1} / {args.epoch}]",
              device)

        top1, top5 = valid(model,
                           valid_loader,
                           criterion,
                           device)

        print(f"top1 : {top1} / top5 : {top5}")

        if top1 > best_top1:
            best_top1 = top1

            torch.save({'state_dict': model.state_dict()},
                       f"best_{args.save_model}.pth")

    # save model after training
    torch.save({'state_dict': model.state_dict()},
               f"{args.save_model}.pth")

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
