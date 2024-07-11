#user-defined libraries.
from config import get_config
from solver import Solver
from utils.tools import getPathList, getSEED4Data

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import numpy as np
import torch
import os

class CusEEGDataset(torch.utils.data.Dataset):
    def __init__(self, train_config, feature_list, label_list):
        self.train_config = train_config
        self.feature_list = feature_list
        self.label_list = label_list

    def __getitem__(self, index):
        if self.train_config.model_type=='ConTL':
            self.feature_list[index]=np.reshape(self.feature_list[index], (self.feature_list[index].shape[0]* self.feature_list[index].shape[1]))
           
            if self.train_config.data_choice == 'deap':
                self.feature_list[index] = np.resize(self.feature_list[index], [310,])        
            self.feature_list[index] = np.expand_dims(self.feature_list[index], axis=0)   
        
        feature = torch.from_numpy(self.feature_list[index]).float()
        
        label = torch.from_numpy(np.asarray(self.label_list[index])).long()
   
        return feature, label

    def __len__(self):
        return len(self.label_list)


def train(X_train, X_val, X_test, y_train, y_val, y_test):

    train_data = CusEEGDataset(train_config, X_train, y_train)
    val_data = CusEEGDataset(train_config, X_val, y_val)
    test_data = CusEEGDataset(train_config, X_test, y_test)

    train_data_loader = DataLoader(train_data, batch_size=train_config.batch_size, shuffle=False)
    val_data_loader = DataLoader(val_data, batch_size = train_config.batch_size, shuffle = False)
    test_data_loader = DataLoader(test_data, batch_size=train_config.batch_size, shuffle=False)

    solver = Solver
    solver = solver(train_config, train_data_loader, val_data_loader, test_data_loader)
    
    solver.build()
    solver.train()
    test_loss, test_acc, test_f1_score=solver.evaluate(test_data_loader, is_load=True)
    print('test_loss: {:.3f} | test_acc: {:.3f} | test_f1_score: {:.3f}'.format(test_loss, test_acc, test_f1_score))

    return test_acc, test_f1_score

def printResults(args, acc_avg, acc_std, f1_avg, f1_std):
    print(f'=======Data choice: {args.data_choice}  | {args.model_type}_result=======')
    print('subject_test_acc_mean:', acc_avg)
    print('subject_test_acc_std:', acc_std)

    print('subject_test_f1_avg:', f1_avg)
    print('subject_test_f1_std:', f1_std)

def saveResults(args, acc_avg, acc_std, f1_avg, f1_std):
    f=open(f'./result/{args.save_file_name}', args.w_mode)
    if args.w_mode=='w':
        f.write('LSTM,')
        f.write('label_type,')
        f.write('lstm_hidden_units,')
        f.write('model,')
        f.write('classification accuracy,')
        f.write('acc std,')
        f.write('F1,')
        f.write('F1 std,')
        f.write('data,')
        f.write('n_ranges,')
        f.write('\n')
    f.write(str(args.lstm)+',')
    f.write(args.label_type+',')
    f.write(str(args.lstm_hidden_size)+',')
    f.write(args.model_type+',')
    f.write(str(round(acc_avg,2))+',')
    f.write(str(round(acc_std,2))+',')
    f.write(str(round(f1_avg,2))+',')
    f.write(str(round(f1_std,2))+',')
    f.write(args.data_choice+',')
    f.write(str(args.n_classes)+'\n')

    f.close()

def saveDEAPResults(args, acc_val, f1_val):
    f=open(f'./result/band-{args.band}_result.csv', args.w_mode)
    if args.w_mode=='w':
        f.write('out_size,')
        f.write('model,')
        f.write('modulator,')
        f.write('classification accuracy,')
        f.write('data')
        f.write('\n')
    f.write(str(args.out_size)+',')
    f.write(args.model_type+',')
    f.write(str(args.modulator)+',')
    f.write(str(round(acc_val, 2))+',')
    f.write(args.data_choice+'\n')
    f.close()

def calcAccMeanStd(acc_list, f1_list):

    acc_avg=np.mean(acc_list)
    acc_std=np.std(acc_list)

    f1_avg=np.mean(f1_list)
    f1_std=np.std(f1_list)

    return acc_avg, acc_std, f1_avg, f1_std

def get_subject_file_map(filelist):
    file_map = {}

    for path in filelist:
        splits = path.split(os.path.sep)
        fn=splits[-1]
        current_subject = fn.split('_')[0]

        if current_subject not in file_map:
            file_map[current_subject] = []
        file_map[current_subject].append(path)
    
    return file_map

def write_cls_accs_for_total_subjects(acc_list, config, savepath):
    with open(savepath, config.w_mode) as f:
        if config.w_mode=='w':
            f.write('Subjects,')
            for i in range(1, len(acc_list)+1):
                f.write(str(i))
                if not i==len(acc_list):
                    f.write(',')
            f.write('\n')
        f.write(config.model_type+',')
        for i in range(len(acc_list)):
            f.write(str(round(acc_list[i],2)))
            if i!=(len(acc_list)-1):
                f.write(',')
        f.write('\n')
        f.close()

def run(train_config):
    if not os.path.exists('result'):
        os.mkdir('result')

    test_acc_list=[]
    test_f1_list=[]
    
    if train_config.data_choice =='deap':
        n_total_subjects=32
        savePreprocessedDeapData(train_config, "./1D_dataset/")
        save3Ddataset(train_config, "./1D_dataset/")

    elif train_config.data_choice=='4': #seed-IV:
        n_total_subjects=15
        train_config.all_files = getPathList(train_config.data_path)
        file_map = get_subject_file_map(train_config.all_files)
                
    for su in range(1, n_total_subjects+1):
        print(f'================Subject{su}===================')
        train_config.subject=str(su)

        if train_config.data_choice=='deap':
            train_x, test_x, train_y, test_y= getDeapData(train_config)    
        elif train_config.data_choice=='4':
            train_x, test_x, train_y, test_y = getSEED4Data(file_map[train_config.subject])
        
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=7)
        
        test_acc, test_f1_score = train(train_x, val_x, test_x, train_y, val_y, test_y)
        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1_score)

    #write classification accuracies for 15 subjects
    write_cls_accs_for_total_subjects(test_acc_list, train_config, f'./result/subject_wise_accuracy_for {train_config.data_choice} data.csv')

    subject_test_acc_avg, subject_test_acc_std, subject_test_f1_avg, subject_test_f1_std = calcAccMeanStd(test_acc_list, test_f1_list)
    printResults(train_config, subject_test_acc_avg, subject_test_acc_std, subject_test_f1_avg, subject_test_f1_std)  
    saveResults(train_config, subject_test_acc_avg, subject_test_acc_std, subject_test_f1_avg, subject_test_f1_std)
            
if __name__=='__main__':
    SEED=336
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False
    np.random.seed(SEED)
    
    train_config = get_config(mode='train')
    
    run(train_config)

        










