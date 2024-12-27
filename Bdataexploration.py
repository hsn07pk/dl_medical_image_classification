import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import models

df = pd.read_csv("./Bdataset/trainLabels.csv")
# print(df.head())

stage_num , stage_frequency = np.unique(df.level , return_counts = True)
# print(stage_num)
# print(stage_frequency)
'''#Results___
#[0 1 2 3 4]
# [25810  2443  5292   873   708] '''


mapp_stage_to_frequency = dict(zip(stage_num , stage_frequency))
# print(mapp_stage_to_frequency)
# {0: 25810, 1: 2443, 2: 5292, 3: 873, 4: 708}


min_stage = sorted(mapp_stage_to_frequency.items() , key = lambda x : x[1])[0]
# print(min_stage)
# (4, 708)

balanced_df = df.groupby('level').apply(lambda x : x.sample(n = min_stage[1] , 
                                                replace = False , 
                                                random_state = 42)).reset_index(drop = True)

# print(balanced_df.head(-5))

'''3530   30297_left      4
3531   8462_right      4
3532   38325_left      4
3533     986_left      4
3534   43997_left      4'''

# print(np.unique(balanced_df.level , return_counts = True))
''' (array([0, 1, 2, 3, 4]), array([708, 708, 708, 708, 708])) '''














