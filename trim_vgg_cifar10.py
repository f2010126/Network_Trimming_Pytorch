import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from vgg import vgg16
from apoz import APoZ
from helper import save_pkl, load_pkl, valid, load_cifar10_data
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
    model = vgg16(pretrained=False, n_class=10).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])

    # show_summary(model) # what is this one?
    summary(model, (3, 32, 32))
    # save apoz pkl or load a new one
    if not os.path.exists(args.apoz_path):
        apoz_obj =APoZ(model)
        apoz = apoz_obj.get_apoz(valid_loader, criterion, args.device)
        save_pkl(apoz, args.apoz_path)
        apoz_obj.deregister()

    else:
        apoz = load_pkl(args.apoz_path)

    # info display current apoz
    print("Average Percentage Of Zero Mean")
    for n, p in zip(module_name, apoz):
        print(f"{n} : {p.mean() * 100 : .2f}%")

    # Masking
    mask = []
    # mask here, ie see what neurons to prune
    for i, p in enumerate(apoz[-3:-1]):
        sorted_arg = np.argsort(p)
        mask.append(sorted_arg < select_rate[i])

    # Conv 5-3 [output]
    model.features[-3] = conv_post_mask(model.features[-3], mask[0], args.device)
    # FC 6 [input, output]
    model.classifier[0] = linear_mask(model.classifier[0], mask[0], mask[1], device=args.device)
    # FC 7 [input]
    model.classifier[3] = linear_pre_mask(model.classifier[3], mask[1], device=args.device)

    # display the pruned model
    summary(model, (3, 32, 32))
    # save the pruned model
    torch.save({'cfg': ['Conv 5-3', 'FC 6'],
                'mask': mask,
                'state_dict': model.state_dict()},
               f"{itr}_{args.save_model}.pth")

    # valid
    acc_top1, acc_top5 = valid(model, valid_loader, criterion, args.device)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Post Trimming Validation for rate {itr} \n"
          f"Acc@1: {acc_top1} \n"
          f"Acc@5: {acc_top5} \n Completed Trimming in "
          "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cifar10 VGG Network Trimming')
    parser.add_argument('--load_path', '-l', type=str,
                        help='Path to load fully pretrained model',
                        default='best_vgg16_full_train_model.pth')
    parser.add_argument('--save_model', type=str, default='apoz_vgg_cifar_model',
                        help='Path to save final apoz model')
    parser.add_argument('--apoz_path', type=str, default='vgg_apoz_fc.pkl',
                        help='Path to apoz values as pkl')
    parser.add_argument('--batch_size', type=int, default=10)

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Trimming VGG16 with Cifar10 with {args}")

    rate = [(488, 3477),
            (451, 2937),
            (430, 2479),
            (420, 2121),
            (400, 1787),
            (390, 1513)]

    # train/valid dataset
    _, loader_valid, _ = load_cifar10_data(batch=128)

    criterion = nn.CrossEntropyLoss().to(device)
    for ctr, one_rate in enumerate(rate):
        trim_network(args, loader_valid, criterion, one_rate, ctr)
