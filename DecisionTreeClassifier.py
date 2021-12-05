import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer, recall_score, accuracy_score
import joblib
import pandas as pd
import argparse
from argparse import RawTextHelpFormatter
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
#import sys
#sys.setrecursionlimit(3000)

def my_custom_score(y_true, y_pred):
    kappa = cohen_kappa_score(y_true, y_pred)
    return kappa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-f', "--feature_num", type=int, help="which version of feature you want to use?", default=0)
    args = parser.parse_args()
    
    new_all_data_path = './data/new_all_data.pickle'
    f = open(new_all_data_path, 'rb') 
    new_all_data_dict = pickle.load(f)
    features_key_list_train = ['no_remove_train_feature', 'remove_low_importance_and_high_correlation_train_feature', 'remove_low_importance_train_feature', 'remove_zero_importance_train_feature']
    features_key_list_test = ['no_remove_test_feature', 'remove_low_importance_and_high_correlation_test_feature', 'remove_low_importance_test_feature', 'remove_zero_importance_test_feature']
    train_data = new_all_data_dict['train']['feature'][features_key_list_train[args.feature_num]]
    train_label = new_all_data_dict['train']['label']
    test_data = new_all_data_dict['test']['feature'][features_key_list_test[args.feature_num]]
    
    oversample = RandomOverSampler(random_state=100)
    train_data_upsample, train_label_upsample = oversample.fit_resample(train_data, train_label)
    
    clf = DecisionTreeClassifier(random_state=100)
    scorer = make_scorer(my_custom_score, greater_is_better=True)
    params_space = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'min_samples_split': [2,3,4,5,6,7,8,9,10],
        'min_samples_leaf': range(1,11,1),
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'ccp_alpha': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        'class_weight': ['balanced', None]
        }
    
    s_CV = RandomizedSearchCV(clf, params_space, cv=5, verbose=1, n_jobs=-1, n_iter=40, scoring=scorer, refit=True, random_state=123)
    s_CV.fit(train_data_upsample, train_label_upsample)
    CV_result = s_CV.cv_results_
    best_clf = s_CV.best_estimator_
    train_pred = best_clf.predict(train_data)
    
    with open('./output/DTC/DTC_upsample_all_result.txt', 'a') as f:
        print('=====================', features_key_list_train[args.feature_num], file=f)
        print('Train Kappa =', round(cohen_kappa_score(train_label, train_pred), 4), file=f)
        print('Train UAR =', round(recall_score(train_label, train_pred, average='macro')*100, 2), '%', file=f)
        print('Train ACC =', round(accuracy_score(train_label, train_pred)*100, 2), '%', file=f)
    #joblib.dump(best_clf, './model/' + features_key_list_train[0] + '/randomforest.model')
    
    test_path = './data/test.csv'
    t_f = pd.read_csv(test_path)
    test_pred = best_clf.predict(test_data)
    ID_arr = []
    for i in range(0, len(t_f['Station'].to_numpy()), 1):
        ID_arr.append(str(t_f['Station'].to_numpy()[i]) + '_' + str(t_f['Season'].to_numpy()[i]))
    ID_arr = np.array(ID_arr)
    combine = np.concatenate((ID_arr[:,np.newaxis], test_pred[:,np.newaxis]), axis=1)
    df = pd.DataFrame(combine, columns = ['ID','LEVEL'])
    out_csv_name = './output/DTC/DTC_upsample' + str(args.feature_num) + '.csv'
    df.to_csv(out_csv_name, index=False)
    