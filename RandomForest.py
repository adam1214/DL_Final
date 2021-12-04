import pickle
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer, recall_score, accuracy_score
import joblib
import pandas as pd

def my_custom_score(y_true, y_pred):
    kappa = cohen_kappa_score(y_true, y_pred)
    return kappa

if __name__ == "__main__":
    new_all_data_path = './data/new_all_data.pickle'
    f = open(new_all_data_path, 'rb') 
    new_all_data_dict = pickle.load(f)
    features_key_list_train = ['no_remove_train_feature', 'remove_low_importance_and_high_correlation_train_feature', 'remove_low_importance_train_feature', 'remove_zero_importance_train_feature']
    features_key_list_test = ['no_remove_test_feature', 'remove_low_importance_and_high_correlation_test_feature', 'remove_low_importance_test_feature', 'remove_zero_importance_test_feature']
    train_data = new_all_data_dict['train']['feature'][features_key_list_train[0]]
    train_label = new_all_data_dict['train']['label']
    test_data = new_all_data_dict['test']['feature'][features_key_list_test[0]]
    
    clf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=123)
    scorer = make_scorer(my_custom_score, greater_is_better=True)
    params_space = {
        'n_estimators': range(10, 501, 10),
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': range(1, 11, 1),
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'oob_score': [True, False],
        'warm_start': [True, False],
        'class_weight': ['balanced', 'balanced_subsample'],
        'ccp_alpha': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        }
    
    s_CV = RandomizedSearchCV(clf, params_space, cv=5, verbose=1, n_jobs=-1, n_iter=100, scoring=scorer, refit=True, random_state=123)
    s_CV.fit(train_data, train_label)
    
    best_clf = s_CV.best_estimator_
    train_pred = best_clf.predict(train_data)
    print('Train Kappa =', round(cohen_kappa_score(train_label, train_pred), 4))
    print('Train UAR =', round(recall_score(train_label, train_pred, average='macro')*100, 2), '%')
    print('Train ACC =', round(accuracy_score(train_label, train_pred)*100, 2), '%')
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
    df.to_csv('output.csv', index=False)
    