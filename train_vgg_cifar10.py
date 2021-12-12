
import time
import torch
import argparse
import torch.nn as nn
from vgg import vgg16
from helper import valid, train, load_cifar10_data
from torchsummary import summary


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='Full training the VGG16 for Cifar10')
    parser.add_argument('--data_path', type=str, default='./',
                        help='Path to root dataset folder ')
    parser.add_argument('--save_model', type=str, default='vgg16_full_train_model',
                        help='Path to model save')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=1)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    # of training hyperparameters in experiments: {base-lr: 0.001, gamma: 0.1, step-size: 3000}
    # gives the loaders
    print(f"Training VGG16 with {args}")
    train_loader, valid_loader, test_loader = load_cifar10_data(batch=args.batch_size)

    model = vgg16(pretrained=False, n_class=10).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                weight_decay=1e-4)

    # show_summary(model) # what is this one?
    summary(model, (3, 32, 32))

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

        print(f"EPOCH {e + 1}/{args.epoch} top1 : {top1} / top5 : {top5}")

        if top1 > best_top1:
            best_top1 = top1

            torch.save({'state_dict': model.state_dict()},
                       f"best_{args.save_model}.pth")

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in "
          "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    # load and test with best model
    checkpoint = torch.load(f"best_{args.save_model}.pth")
    model = vgg16(pretrained=False, n_class=10).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    start = time.time()
    top1, top5 = valid(model,
                       test_loader,
                       criterion,
                       device)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Best Model on test set " 
          f"top1 : {top1} / top5 : {top5} \n Inference completed in"
          "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
