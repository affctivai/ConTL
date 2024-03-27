from sklearn import preprocessing
from scipy.signal import butter, lfilter

import math
import scipy.io as scio
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import scipy.signal as signal
from sklearn.model_selection import train_test_split

def getPathList(root):
    items = os.listdir(root)
    pathList=[]
    for item in items:
        full_path = os.path.join(root, item)
        if os.path.isfile(full_path):
            pathList.append(full_path)
        else:
            pathList.extend(getPathList(full_path))
    return pathList


def getSEED4Data(pathList):

    session1_label = ['1','2','3','0','2','0','0','1','0','1','2','1','1','1','2','3','2','2','3','3','0','3','0','3']

    session2_label = ['2','1','3','0','0','2','0','2','3','3','2','3','2','0','1','1','2','1','0','3','0','1','3','1']
    session3_label = ['1','2','2','1','3','3','3','1','1','2','1','0','2','3','3','0','2','3','0','0','2','0','1','0']

    session_label_dict={}
    session_label_dict['1']=session1_label
    session_label_dict['2']=session2_label
    session_label_dict['3']=session3_label

    train_features=[]
    train_labels=[]
    test_features=[]
    test_labels=[]

    for path in pathList:

        all_trials_dict = scio.loadmat(path)

        experiment_name = path.split(os.path.sep)[-1]

        subject_name=experiment_name.split('_')[0]

        session_type=path.split(os.path.sep)[-2]

        for i in range(1, 25):
            key=f'de_LDS{i}'
            de_features = all_trials_dict[key]
            de_features=de_features.transpose(1, 0, 2)
            # de_features=de_features.reshape(-1, 310)
            label=int(session_label_dict[session_type][i-1])

            if i<17:
                train_features.extend(de_features)
                train_labels.extend([label]*de_features.shape[0])
            else:
                test_features.extend(de_features)
                test_labels.extend([label]*de_features.shape[0])

    train_features = np.asarray(train_features)
    test_features = np.asarray(test_features)
    target_mean0 = np.mean(train_features[:,:,0])
    target_std0 = np.std(train_features[:,:,0])
    target_mean1 = np.mean(train_features[:,:,1])
    target_std1 = np.std(train_features[:,:,1])
    target_mean2 = np.mean(train_features[:,:,2])
    target_std2 = np.std(train_features[:,:,2])
    target_mean3 = np.mean(train_features[:,:,3])
    target_std3 = np.std(train_features[:,:,3])
    target_mean4 = np.mean(train_features[:,:,4])
    target_std4 = np.std(train_features[:,:,4])

    train_features[:,:,0] = (train_features[:,:,0]-target_mean0)/target_std0
    train_features[:,:,1] = (train_features[:,:,1]-target_mean1)/target_std1
    train_features[:,:,2] = (train_features[:,:,2]-target_mean2)/target_std2
    train_features[:,:,3] = (train_features[:,:,3]-target_mean3)/target_std3
    train_features[:,:,4] = (train_features[:,:,4]-target_mean4)/target_std4

    test_features[:,:,0] = (test_features[:,:,0]-target_mean0)/target_std0
    test_features[:,:,1] = (test_features[:,:,1]-target_mean1)/target_std1
    test_features[:,:,2] = (test_features[:,:,2]-target_mean2)/target_std2
    test_features[:,:,3] = (test_features[:,:,3]-target_mean3)/target_std3
    test_features[:,:,4] = (test_features[:,:,4]-target_mean4)/target_std4

    train_features = list(train_features)
    test_features = list(test_features)

    return train_features, test_features, train_labels, test_labels
