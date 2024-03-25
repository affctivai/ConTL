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

def read_file(file):
    file = scio.loadmat(file)
    trial_data = file['data']
    
    return trial_data, file["arousal_labels"], file["valence_labels"]

def pre_process(path):
    # DE feature vector dimension of each band
    data_3D = np.empty([0,9,9])
    sub_vector_len = 32
    trial_data, arousal_labels, valence_labels = read_file(path)

    #data = preprocessing.scale(trial_data,axis=1, with_mean=True,with_std=True,copy=True)

    return trial_data, arousal_labels,valence_labels

def compute_DE(signal):
    variance = np.var(signal,ddof=1)
    return math.log(2*math.pi*math.e*variance)/2

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def decompose(file):
    data = scio.loadmat(file)['data']
    frequency = 128

    decomposed_de = np.empty([0,4,60])

    for trial in range(40):
        temp_de = np.empty([0,60])

        for channel in range(32):
            trial_signal = data[trial,channel,384:]

            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

            DE_theta = np.zeros(shape=[0],dtype = float)
            DE_alpha = np.zeros(shape=[0],dtype = float)
            DE_beta =  np.zeros(shape=[0],dtype = float)
            DE_gamma = np.zeros(shape=[0],dtype = float)

            for index in range(60):
                DE_theta =np.append(DE_theta,compute_DE(theta[index*frequency:(index+1)*frequency]))
                DE_alpha =np.append(DE_alpha,compute_DE(alpha[index*frequency:(index+1)*frequency]))
                DE_beta =np.append(DE_beta,compute_DE(beta[index*frequency:(index+1)*frequency]))
                DE_gamma =np.append(DE_gamma,compute_DE(gamma[index*frequency:(index+1)*frequency]))
            temp_de = np.vstack([temp_de,DE_theta])
            temp_de = np.vstack([temp_de,DE_alpha])
            temp_de = np.vstack([temp_de,DE_beta])
            temp_de = np.vstack([temp_de,DE_gamma])
        temp_trial_de = temp_de.reshape(-1,4,60)
        decomposed_de = np.vstack([decomposed_de,temp_trial_de])

    de_features = decomposed_de.reshape(-1,32,4,60).transpose([0,3,2,1]).reshape(-1,4,32)

    print("trial_DE shape:",decomposed_de.shape)
    return de_features

def setLabel(degree, n_ranges):

    if n_ranges==9:
        label = np.round(degree)-1
    else:
        offset = 9/n_ranges

        upper=1+offset
        if degree <=upper:
            label=int(0)

        for i in range(1,n_ranges):
            lower=upper
            upper+=offset
            if lower<degree<=upper:
                label = int(i)
                break

    return label

def get_labels(file, n_ranges):
    #0 valence, 1 arousal, 2 dominance, 3 liking

    valence_labels = scio.loadmat(file)["labels"][:,0]
    arousal_labels = scio.loadmat(file)["labels"][:,1]

    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    for i in range(len(valence_labels)):
        for j in range(0,60):
            valence_label = setLabel(valence_labels[i], n_ranges)

            arousal_label = setLabel(arousal_labels[i], n_ranges)
            final_valence_labels = np.append(final_valence_labels, valence_label)
            final_arousal_labels = np.append(final_arousal_labels, arousal_label)

    return final_arousal_labels,final_valence_labels

def savePreprocessedDeapData(train_config, oneD_dataset_dir):
    raw_dataset_dir = train_config.data_path

    if os.path.isdir(oneD_dataset_dir)==False:
        os.makedirs(oneD_dataset_dir)
        #save preprocessed deap data.
        for i in range(2,10):
            n_ranges=i
            dir_name=f'range_{n_ranges}_labels/'

            for file in os.listdir(raw_dataset_dir):
                file_path = os.path.join(raw_dataset_dir,file)
                trial_DE = decompose(file_path)
                arousal_labels,valence_labels = get_labels(file_path, n_ranges)
                current_result_dir=os.path.join(oneD_dataset_dir, dir_name)
                if not os.path.exists(current_result_dir):
                    os.mkdir(current_result_dir)
                scio.savemat(current_result_dir+"DE_"+file,{"data":trial_DE,"valence_labels":valence_labels,"arousal_labels":arousal_labels})

def save3Ddataset(train_config, oneD_dataset_dir):
    output_dir = "./3D_dataset/"

    if os.path.isdir(output_dir)==False:
        os.makedirs(output_dir)

    n_ranges=train_config.n_classes
    current_source_dir = os.path.join(oneD_dataset_dir, f'range_{n_ranges}_labels')

    for file in os.listdir(current_source_dir):
        file_path = os.path.join(current_source_dir, file)
        data, arousal_labels,valence_labels = pre_process(file_path)
        scio.savemat(output_dir+file,{"data":data,"valence_labels":valence_labels,"arousal_labels":arousal_labels})


def getDeapData(train_config):
    dataset_dir = "./3D_dataset/DE_"
    input_file='s'+str(train_config.subject)
    su=int(train_config.subject)
    if su <10:
        input_file='s0'+str(su)

    print("loading ",dataset_dir+input_file,".mat")
    data_file = scio.loadmat(dataset_dir+input_file+".mat")

    datasets = data_file["data"]
    
    datasets=datasets.transpose(0,2,1)
  
    label_key = train_config.label_type+"_labels"
    labels = data_file[label_key]

    #2018-5-16 modified
    time_step = 1
    label_index = [i for i in range(0,labels.shape[1],time_step)]

    labels = labels[0,[label_index]]
    labels = np.squeeze(np.transpose(labels))
    
    train_x, test_x, train_y, test_y = \
                            train_test_split(datasets, labels, test_size=0.3, random_state=7)
    
    #z-score
    target_mean0 = np.mean(train_x[:,:,0])
    target_std0 = np.std(train_x[:,:,0])
    target_mean1 = np.mean(train_x[:,:,1])
    target_std1 = np.std(train_x[:,:,1])
    target_mean2 = np.mean(train_x[:,:,2])
    target_std2 = np.std(train_x[:,:,2])
    target_mean3 = np.mean(train_x[:,:,3])
    target_std3 = np.std(train_x[:,:,3])

    train_x[:,:,0] = (train_x[:,:,0]-target_mean0)/target_std0
    train_x[:,:,1] = (train_x[:,:,1]-target_mean1)/target_std1
    train_x[:,:,2] = (train_x[:,:,2]-target_mean2)/target_std2
    train_x[:,:,3] = (train_x[:,:,3]-target_mean3)/target_std3

    test_x[:,:,0] = (test_x[:,:,0]-target_mean0)/target_std0
    test_x[:,:,1] = (test_x[:,:,1]-target_mean1)/target_std1
    test_x[:,:,2] = (test_x[:,:,2]-target_mean2)/target_std2
    test_x[:,:,3] = (test_x[:,:,3]-target_mean3)/target_std3
    
    train_x = list(train_x)
    test_x = list(test_x)
    
    return train_x, test_x, train_y, test_y

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
