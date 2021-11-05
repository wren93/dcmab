from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def preprocess(dataset_path, test_size):
    df = pd.read_csv(dataset_path)
    mushrooms = pd.get_dummies(df)

    # divide training feature and label
    LABELS = ['class_e', 'class_p']
    FEATURES = [a for a in mushrooms.columns if a not in LABELS]

    # binary reward - 1 if edible, 0 otherwise
    y = mushrooms[LABELS[0]]
    # feature vectors
    x = mushrooms[FEATURES]
    num_input = x.shape[1] + 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test, num_input


class MushroomDataset(Dataset):
    def __init__(self, data):
        super(MushroomDataset, self).__init__()
        self.data_x = np.array(data[0])
        self.data_y = np.array(data[1])

    def __getitem__(self, index):
        item = self.data_x[index, :]
        # mushroom dataset input size: 118
        item_eat = torch.tensor(np.concatenate((item, 1), axis=None), dtype=torch.float32)
        item_neat = torch.tensor(np.concatenate((item, 0), axis=None), dtype=torch.float32)
        y_eat = torch.tensor([self.data_y[index]], dtype=torch.float32)
        y_neat = torch.tensor(np.random.randint(2, size=1), dtype=torch.float32)
        ret = {'eat_x': item_eat, 'eat_y': y_eat, 'neat_x': item_neat, 'neat_y': y_neat}
        return ret

    def __len__(self):
        return self.data_y.shape[0]