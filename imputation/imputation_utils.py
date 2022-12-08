import pandas as pd
from mTAN import *
from toy_dataset import data_utils

import PyPOTS.pypots

def load_toy_dataset(name=None):
    if name is None:
        name = 'toydataset_small'
    path = data_utils.datasets_dict[name]['path']
    data_utils.ToyDataset(path=path)
    data_utils.get_Toy_Dataloader()

print(('Done!'))


