import time
import copy
import torch
import torch.nn as nn
import argparse
from PIL import ImageFile

from vgg import vgg16
from helper import train, valid, load_cifar10_data
from converter import conv_post_mask, linear_mask, linear_pre_mask

# setting
ImageFile.LOAD_TRUNCATED_IMAGES = True


def setup_trimmed_model(model_name, device):
    # load model, trim acc to mask and return the model and mask used
    checkpoint = torch.load(model_name)
    trim_model = vgg16(pretrained=False, n_class=10).to(device)
    trim_mask = checkpoint['mask']
    # New layers being added to default model as per the apoz trimming
    # Conv 5-3 [output]
    trim_model.features[-3] = conv_post_mask(trim_model.features[-3], trim_mask[0])
    # FC 6 [input, output]
    trim_model.classifier[0] = linear_mask(trim_model.classifier[0], trim_mask[0], trim_mask[1])
    # FC 7 [input]
    trim_model.classifier[3] = linear_pre_mask(trim_model.classifier[3], trim_mask[1])
    trim_model.load_state_dict(copy.deepcopy(checkpoint['state_dict']))

    return trim_model, trim_mask


def finetune_network(args, load_train, load_test, load_valid, ctr):
    # load prune model
    model, mask = setup_trimmed_model(f"{ctr}_apoz_vgg_cifar_model.pth", args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                weight_decay=1e-4)

    best_top1 = 0
    for e in range(args.epoch):
        train(model,
              load_train,
              criterion,
              optimizer,
              f"EPOCH : [{e + 1} / {args.epoch}]",
              args.device)

        top1, top5 = valid(model,
                           load_valid,
                           criterion,
                           args.device)

        # print(f" TRAINING EPOCH {e + 1} / {args.epoch} top1 : {top1} / top5 : {top5}")

        if top1 > best_top1:
            best_top1 = top1
            # save the mask as well for later testing
            torch.save({'cfg': ['Conv 5-3', 'FC 6'],
                        'mask': mask,
                        'state_dict': model.state_dict()},
                       f"{ctr}_best_{args.save_model}.pth")

    # load and test best for each
    model, _ = setup_trimmed_model(f"{ctr}_best_{args.save_model}.pth", args.device)
    # measure inference time
    start = time.time()
    top1, top5 = valid(model,
                       load_test,
                       criterion,
                       device)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Fine tuned apoz model for rate {ctr} on test set."
          f"top1 : {top1} / top5 : {top5} \nInference time: "
          "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGG Cifar10 Network FineTuning')
    parser.add_argument('--save_model', type=str, default='apoz_finetune_vgg',
                        help='Path to model save')
    parser.add_argument('--prune_path', '-p', type=str,
                        help="Name of apoz trimmed model",
                        default='apoz_vgg_cifar_model')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Fine Tuning VGG16 with Cifar10 with {args}")
    load_trn, load_val, load_tst = load_cifar10_data(batch=args.batch_size)
    # fine tune all 6.
    # TODO: make it iterartive
    for i in range(6):
        finetune_network(args, load_trn, load_tst, load_val, i)
