# dataset: https://www.kaggle.com/aavigan/uci-mushroom-data/data
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
from scipy.stats import beta as beta_dist

from dataset import get_dataloader
from models import DCMABNet


# deep contextual multi-armed bandits
def dcmab(opt):
    trainloader, testloader, num_input = get_dataloader(opt)
    
    # get network
    model = DCMABNet(n_input=num_input, n_hidden=opt.num_hidden, dropout_rate=opt.dropout_rate)
    model.to(opt.device)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # define loss function
    loss_func = nn.BCEWithLogitsLoss()

    cumulative_regret = 0
    regret_curve = []

    # training
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
                    val_eat = torch.sigmoid(model(eat_x[i, :]))
                    val_neat = torch.sigmoid(model(neat_x[i, :]))

                    # compute regret
                    oracle = max(eat_y[i, :], neat_y[i, :])

                    if val_eat > val_neat:
                        cumulative_regret += int(oracle - eat_y[i, :])
                    else:
                        cumulative_regret += int(oracle - neat_y[i, :])

                    regret_curve.append(cumulative_regret)

            running_loss = 0
            # network forward & computing loss
            for i in range(2):
                # TODO: change training logic: only the observed data can be used to train the model
                res = model(data[i])

                loss = loss_func(res, label[i])
                running_loss += loss.item()

                # back-propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
        print("Epoch {}: running loss = {}".format(epoch + 1, running_loss))
    
    # TODO: test accuracy
    
    return regret_curve


# non-contextual multi-armed bandits: beta-bernoulli bandits with Thompson sampling
def ncts(opt):
    trainloader, testloader, num_input = get_dataloader(opt)

    # beta distribution parameters
    # prior distribution: beta(1, 1) (uniform distribution)
    # for both arm
    beta_param_eat = [0, 0]
    beta_param_neat = [0, 0]

    regret_curve = []
    cumulative_regret = 0

    for epoch in range(opt.epoch):
        for iter_id, batch in enumerate(trainloader):
            # get data
            eat_y = batch['eat_y'].squeeze(0)
            neat_y = batch['neat_y'].squeeze(0)

            # sampling from the distributions
            val_eat = np.random.beta(1 + beta_param_eat[0], 1 + beta_param_eat[1])
            val_neat = np.random.beta(1 + beta_param_neat[0], 1 + beta_param_neat[1])

            oracle = max(eat_y, neat_y)

            if val_eat > val_neat:
                cumulative_regret += int(oracle - eat_y)
                beta_param_eat[1 - int(eat_y)] += 1
            else:
                cumulative_regret += int(oracle - neat_y)
                beta_param_neat[1 - int(neat_y)] += 1

            regret_curve.append(cumulative_regret)
    
    print(beta_param_eat)
    print(beta_param_neat)
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
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_val', type=int, default=64)
    parser.add_argument('--save_path', default='./pretrained')
    parser.add_argument('--output_name', default='two_layer_net_mushroom.pth')

    opt = parser.parse_args()

    """
    comparison between deep contextual multi-armed bandits and 
    non-contextual beta-bernoulli bandits
    """
    # deep contextual multi-armed bandits
    regret_curve_dcmab = dcmab(opt)

    # Beta-Bernoulli bandits
    opt.batch_size_train = 1
    regret_curve_ts = ncts(opt)

    # plot regret curve
    plt.plot(np.arange(1, len(regret_curve_ts) + 1), regret_curve_ts)
    plt.plot(np.arange(1, len(regret_curve_dcmab) + 1), regret_curve_dcmab)
    plt.legend(["ts", "dcmab"])
    plt.show()

    """
    comparison between different dropout rate for deep 
    contextual multi-armed bandits
    """
    dropout_rates = [0.1, 0.2, 0.4, 0.6, 0.8]
    regret_curves = []
    for i in range(len(dropout_rates)):
        opt.dropout_rate = dropout_rates[i]
        regret_curve_dcmab = dcmab(opt)
        regret_curves.append(regret_curve_dcmab)
        plt.plot(np.arange(1, len(regret_curve_dcmab) + 1), regret_curve_dcmab)
    plt.legend(["dropout rate = {}".format(j) for j in dropout_rates])
    plt.show()