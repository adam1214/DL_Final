import pickle
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer, recall_score, accuracy_score
import joblib
import pandas as pd
import argparse
from argparse import RawTextHelpFormatter
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pdb
from collections import Counter
#import sys
#sys.setrecursionlimit(3000)

class train_val_Data(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        y_data -= 1
        #self.y_data = torch.nn.functional.one_hot(y_data.long())
        self.y_data = y_data.long()
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
class test_Data(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # Number of input features is 10.
        self.layer_1 = nn.Linear(10, 64) 
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, 10) 
        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        self.activate_fun = nn.GELU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.batchnorm3 = nn.BatchNorm1d(16)
        
    def forward(self, inputs):
        x = self.activate_fun(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.activate_fun(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.activate_fun(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-b', "--batch_size", type=int, help="set batch size", default=16)
    parser.add_argument('-l', "--lr", type=float, help="set learning rate", default=0.0001)
    parser.add_argument('-e', "--epoch", type=int, help="set epoch", default=200)
    parser.add_argument('-w', "--weight_decay", type=float, help="set weight_decay", default=0.01)
    parser.add_argument('-f', "--feature_num", type=int, help="which version of feature you want to use?", default=1)
    parser.add_argument('-c', "--weighted_cross_entropy", type=bool, help="set weighted cross_entropy or not", default=True)
    args = parser.parse_args()
    print(args)
    torch.manual_seed(100)
    
    new_all_data_path = './data/new_all_data_after_Fa2.pickle'
    f = open(new_all_data_path, 'rb') 
    new_all_data_dict = pickle.load(f)
    features_key_list_train = ['no_remove_train_feature', 'remove_low_importance_and_high_correlation_train_feature']
    features_key_list_test = ['no_remove_test_feature', 'remove_low_importance_and_high_correlation_test_feature', 'remove_low_importance_test_feature', 'remove_zero_importance_test_feature']
    
    best_clf = joblib.load('./model/' + '/randomforest_' + str(args.feature_num) + '.model')
    train_data = new_all_data_dict['train']['feature'][features_key_list_train[args.feature_num]]
    train_label = new_all_data_dict['train']['label'] # train label for DNN
    test_data = new_all_data_dict['test']['feature'][features_key_list_test[args.feature_num]]
    
    test_pred_prob = best_clf.predict_proba(test_data) # test feature for DNN
    train_pred_prob = best_clf.predict_proba(train_data) # train feature for DNN
    arg_max_train_pred_prob = np.argmax(train_pred_prob, axis=1)
    arg_max_train_pred_prob += 1
    print('Original RF performance#####', features_key_list_train[args.feature_num])
    print('Train Kappa =', round(cohen_kappa_score(train_label, arg_max_train_pred_prob), 4))
    print('Train UAR =', round(recall_score(train_label, arg_max_train_pred_prob, average='macro')*100, 2), '%')
    print('Train ACC =', round(accuracy_score(train_label, arg_max_train_pred_prob)*100, 2), '%')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(train_pred_prob, train_label.to_numpy(), random_state=1234, test_size=0.2, shuffle=True)
    
    train_data = train_val_Data(torch.FloatTensor(train_inputs), torch.FloatTensor(train_outputs))
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    
    val_data = train_val_Data(torch.FloatTensor(val_inputs), torch.FloatTensor(val_outputs))
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    
    test_data = test_Data(torch.FloatTensor(test_pred_prob))
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    
    model = DNN()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_label_counter = Counter(train_outputs)
    max_label_cnt = max(train_label_counter, key=train_label_counter.get)
    
    if args.weighted_cross_entropy == True:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([max_label_cnt/train_label_counter[1], max_label_cnt/train_label_counter[2], max_label_cnt/train_label_counter[3], max_label_cnt/train_label_counter[4], max_label_cnt/train_label_counter[5], max_label_cnt/train_label_counter[6], max_label_cnt/train_label_counter[7], max_label_cnt/train_label_counter[8], max_label_cnt/train_label_counter[9], max_label_cnt/train_label_counter[10]]).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
        
    max_kappa_val = -100
    best_epoch = 0
    for e in range(1, (args.epoch)+1, 1):
        # training
        model.train()
        epoch_loss = 0
        epoch_kappa_train = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            kappa = cohen_kappa_score(y_batch.cpu(), torch.argmax(y_pred, dim=1).cpu())
            loss = criterion(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_kappa_train += kappa.item()
            epoch_loss += loss.item()
        
        # validation
        preds = []
        labels = []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                y_pred = model(X_batch)
                preds += torch.argmax(y_pred, dim=1).tolist()
                labels += y_batch.cpu().tolist()
            kappa_val = cohen_kappa_score(labels, preds)
        
        if kappa_val > max_kappa_val:
            max_kappa_val = kappa_val
            best_epoch = e
            checkpoint = {'epoch': e, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            torch.save(checkpoint, './model/RF_DNN_best_model.pth')
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | train_kappa: {epoch_kappa_train/len(train_loader):.4f} | val_kappa: {kappa_val:.4f}')
    
    print('The best epoch:', best_epoch)
    model = DNN()
    model.to(device)
    checkpoint = torch.load('./model/RF_DNN_best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # check the best model's performance on training data
    preds = []
    labels = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred = model(X_batch)
            preds += torch.argmax(y_pred, dim=1).tolist()
            labels += y_batch.cpu().tolist()
    print('RF -> DNN performance#####')
    print('Train Kappa =', round(cohen_kappa_score(labels, preds), 4))
    print('Train UAR =', round(recall_score(labels, preds, average='macro')*100, 2), '%')
    print('Train ACC =', round(accuracy_score(labels, preds)*100, 2), '%')
    
    # testing
    test_pred = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            
            y_pred = model(X_batch)
            test_pred += torch.argmax(y_pred, dim=1).tolist()
    test_pred = np.asarray(test_pred)
    test_pred += 1
    ID_arr = []
    test_path = './data/test.csv'
    t_f = pd.read_csv(test_path)
    for i in range(0, len(t_f['Station'].to_numpy()), 1):
        ID_arr.append(str(t_f['Station'].to_numpy()[i]) + '_' + str(t_f['Season'].to_numpy()[i]))
    ID_arr = np.array(ID_arr)
    combine = np.concatenate((ID_arr[:,np.newaxis], test_pred[:,np.newaxis]), axis=1)
    df = pd.DataFrame(combine, columns = ['ID','LEVEL'])
    out_csv_name = './output/RF/RF_upsample' + str(args.feature_num) + '.csv'
    df.to_csv(out_csv_name, index=False)