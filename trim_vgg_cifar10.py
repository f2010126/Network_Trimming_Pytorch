import os
import sys
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from vgg import vgg16
from apoz import APoZ
from helper import save_pkl, load_pkl, valid
from converter import conv_post_mask, linear_mask, linear_pre_mask
from torchsummary import summary


def trim_network(args, valid_loader, criterion, select_rate, itr):
    #
    start = time.time()

    module_name = ['Conv 1-1', 'Conv 1-2', 'Conv 2-1', 'Conv 2-2', 'Conv 3-1',
                   'Conv 3-2', 'Conv 3-3', 'Conv 4-1', 'Conv 4-2', 'Conv 4-3',
                   'Conv 5-1', 'Conv 5-2', 'Conv 5-3', 'FC 6', 'FC 7']

    # replace this by our model
    checkpoint = torch.load(args.load_path)
    model = vgg16(pretrained=True).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])

    # show_summary(model) # what is this one?
    summary(model, (3, 224, 224))
    # save apoz pkl or load a new one
    if not os.path.exists(args.apoz_path):
        apoz = APoZ(model).get_apoz(valid_loader, criterion, device)
        save_pkl(apoz, args.apoz_path)

    else:
        apoz = load_pkl(args.apoz_path)

    # info display current apoz
    print("Average Percentage Of Zero Mean")
    for n, p in zip(module_name, apoz):
        print(f"{n} : {p.mean() * 100 : .2f}%")

    # Masking
    mask = []
    # mask here
    for i, p in enumerate(apoz[-3:-1]):
        sorted_arg = np.argsort(p)
        mask.append(sorted_arg < select_rate[i])

    # Conv 5-3 [output]
    model.features[-3] = conv_post_mask(model.features[-3], mask[0], device)
    # FC 6 [input, output]
    model.classifier[0] = linear_mask(model.classifier[0], mask[0], mask[1], device=device)
    # FC 7 [input]
    model.classifier[3] = linear_pre_mask(model.classifier[3], mask[1], device=device)

    # display the pruned model
    summary(model, (3, 224, 224))
    # save the pruned model
    torch.save({'cfg': ['Conv 5-3', 'FC 6'],
                'mask': mask,
                'state_dict': model.state_dict()},
               f"{itr}_{args.save_model}.pth")

    # valid
    acc_top1, acc_top5 = valid(model, valid_loader, criterion)

    print(f"Acc@1: {acc_top1} \n"
          f"Acc@5: {acc_top5} \n")

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cifar10 VGG Network Trimming')
    parser.add_argument('--load_path', '-l', type=str,
                        help='Path to load fully pretrained model',
                        default='vgg16_full_train_model.pth')
    parser.add_argument('--save_model', type=str, default='apoz_vgg_cifar_model',
                        help='Path to save final apoz model')
    parser.add_argument('--apoz_path', type=str, default='vgg_apoz_fc.pkl',
                        help='Path to apoz values as pkl')
    parser.add_argument('--batch_size', type=int, default=10)

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Pruning VGG16 with Cifar10 with {args}")

    rate = [(488, 3477),
            (451, 2937),
            (430, 2479),
            (420, 2121),
            (400, 1787),
            (390, 1513)]

    # train/valid dataset
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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

    train_set, val_set = torch.utils.data.random_split(cifar_dataset, [45000, 5000])

    valid_loader = torch.utils.data.DataLoader(val_set,
                                               batch_size=args.batch_size,
                                               pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)
    for i, one_rate in enumerate(rate):
        trim_network(args, valid_loader, criterion, one_rate, i)
