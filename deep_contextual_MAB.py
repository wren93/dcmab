# dataset: https://www.kaggle.com/aavigan/uci-mushroom-data/data
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import preprocess, MushroomDataset
from model import DCMABNet


# deep contextual multi-armed bandits
def dcmab(opt):
    # get dataset
    x_train, x_test, y_train, y_test, num_input = preprocess(opt.dataset_path, opt.test_size)
    trainset = MushroomDataset((x_train, y_train))
    testset = MushroomDataset((x_test, y_test))

    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=0
        )

    testloader = DataLoader(
        testset,
        batch_size=opt.batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=0
        )
    
    # get network
    model = DCMABNet(n_input=num_input, n_hidden=opt.num_hidden, dropout_rate=opt.dropout_rate)
    model.to(opt.device)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # define loss function
    loss_func = nn.BCEWithLogitsLoss()

    cumulative_regret = 0
    regret_curve = []

    for epoch in range(opt.epoch):
        for iter_id, batch in enumerate(trainloader):
            for item in batch:
                batch[item] = batch[item].to(opt.device)
            
            # contextual variable and reward
            # for eating and not eating the mushroom
            eat_x = batch['eat_x']
            eat_y = batch['eat_y']
            neat_x = batch['neat_x']
            neat_y = batch['neat_y']

            data = [eat_x, neat_x]
            label = [eat_y, neat_y]

            with torch.no_grad():
                # Thompson sampling & regret computation
                for i in range(eat_x.shape[0]):
                    # Thompson sampling
                    reward_eat = torch.sigmoid(model(eat_x[i, :]))
                    reward_neat = torch.sigmoid(model(neat_x[i, :]))

                    # compute regret
                    oracle = max(eat_y[i, :], neat_y[i, :])

                    if reward_eat > reward_neat:
                        cumulative_regret += int(oracle - eat_y[i, :])
                    else:
                        cumulative_regret += int(oracle - neat_y[i, :])

                    regret_curve.append(cumulative_regret)

            running_loss = 0
            # network forward & computing loss
            for i in range(2):
                res = model(data[i])

                loss = loss_func(res, label[i])
                running_loss += loss.item()

                # back-propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
        print("Epoch {}: loss = {}".format(epoch, running_loss))
    
    return regret_curve


# non-contextual multi-armed bandits: beta-bernoulli bandits with Thompson sampling
def ncts(opt):
    regret_curve = []
    return regret_curve
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # torch settings
    parser.add_argument('--seed', type=int, default=223)
    parser.add_argument('--device', type=str, default='cpu',
                        help="'cuda' for GPU training, 'cpu' for CPU training")

    # dataset settings
    parser.add_argument('--dataset_path', default='./mushroom/mushrooms.csv')
    parser.add_argument('--test_size', type=float, default=0.2)

    # network settings
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--dropout_rate', type=float, default=0.2)

    # training settings
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_val', type=int, default=32)
    parser.add_argument('--save_path', default='./pretrained')
    parser.add_argument('--output_name', default='two_layer_net_mushroom.pth')

    opt = parser.parse_args()

    regret_curve_dcmab = dcmab(opt)
    # TODO: Beta-Bernoulli bandits
    regret_curve_ts = ncts(opt)

    # TODO: plot regret curve
    plt.plot(np.arange(1, len(regret_curve_dcmab) + 1), regret_curve_dcmab)
    plt.show()