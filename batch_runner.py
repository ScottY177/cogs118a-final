import pandas as pd
import numpy as np
import joblib
import time

from os import path
from my_package import creat_best_model, eval_model, \
    append_result, parse_to_dataframe, test_train_split_testing_size, send_email_after_finished
from data_cleaning import clean_all_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.svm import SVC  # So inefficient for large dataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

path_name = ['data/adult_clean.csv', 'data/cape_clean.csv', 'data/cover_clean.csv',
             'data/letter_clean_p1.csv', 'data/letter_clean_p2.csv']

if not all([path.exists(name) for name in path_name]):  # only clean dataset if not previously done so.
    clean_all_dataset()

encoded_adult = pd.read_csv("data/adult_clean.csv")
encoded_cape = pd.read_csv("data/cape_clean.csv")
encoded_cover = pd.read_csv("data/cover_clean.csv")
encoded_letter_p1 = pd.read_csv("data/letter_clean_p1.csv")
encoded_letter_p2 = pd.read_csv("data/letter_clean_p2.csv")

# Initialize the Logistical Regression's grid Search #
pipe_logistical = Pipeline([('std', StandardScaler()),
                            ('classifier', LogisticRegression())])

search_space_logistical = [{'classifier': [LogisticRegression(solver='saga', max_iter=5000)],
                            'classifier__penalty': ['l1', 'l2'],
                            'classifier__C': np.logspace(-4, 4, 9)},
                           {'classifier': [LogisticRegression(solver='lbfgs', max_iter=5000)],
                            'classifier__penalty': ['l2'],
                            'classifier__C': np.logspace(-4, 4, 9)},
                           {'classifier': [LogisticRegression(penalty='none', max_iter=5000)],
                            'classifier__solver': ['lbfgs', 'saga']},
                           ]

clf_logistical = GridSearchCV(pipe_logistical, search_space_logistical, scoring=['accuracy', 'roc_auc_ovr', 'f1_micro'],
                              refit=False, verbose=3, n_jobs=-1, pre_dispatch='2*n_jobs')

# Initialize the MLP clf
hidden_units = [1, 2, 4, 8, 32, 128]
momentum = [0, 0.2, 0.5, 0.9, 0.99]
pipe_mlp = Pipeline([('std', StandardScaler()),
                     ('classifier', MLPClassifier())])
search_space_mlp = [
    {
        'classifier': [MLPClassifier(max_iter=5000, learning_rate='adaptive')],
        'classifier__hidden_layer_sizes': [tuple([num]) for num in hidden_units],
        'classifier__momentum': momentum,
        'classifier__alpha': np.logspace(-7, 0, 8)
    }
]

clf_mlp = GridSearchCV(pipe_mlp, search_space_mlp, scoring=['accuracy', 'f1_micro', 'roc_auc_ovr'],
                       refit=False, verbose=3, n_jobs=-1, pre_dispatch='2*n_jobs')

# Initialize the SVM's grid Search #
c_values = np.logspace(2, -7, 10)
gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2]
pipe_svm = Pipeline([('std', StandardScaler()),
                     ('classifier', SVC())])

search_space_svm = [
    {
        'classifier': [SVC(kernel='poly', probability=True, cache_size=2000)],
        'classifier__degree': [2, 3],
        'classifier__C': c_values,
    }, {
        'classifier': [SVC(kernel='rbf', probability=True, cache_size=2000)],
        'classifier__C': c_values,
        'classifier__gamma': gamma
    }, {
        'classifier': [SVC(kernel='linear', probability=True, cache_size=2000)],
        'classifier__C': c_values[:-2]
    }
]

clf_svm = GridSearchCV(pipe_svm, search_space_svm, scoring=['accuracy', 'f1_micro', 'roc_auc_ovr'],
                       refit=False, verbose=3, n_jobs=-1, pre_dispatch='2*n_jobs')

# Initialize the Random Forest's grid Search
pipe_rf = Pipeline([('std', StandardScaler()),
                    ('classifier', RandomForestClassifier())])
feature_size = [1, 2, 4, 6, 8, 12, 16, 20, 'auto', 'log2']
n_estimators = [40, 160, 640, 1024]
max_depth = [10, 50, None]
search_space_rf = [
    {
        'classifier': [RandomForestClassifier(n_jobs=-1)],
        'classifier__max_features': feature_size,
        'classifier__n_estimators': n_estimators,
        'classifier__max_depth': max_depth,
        'classifier__max_leaf_nodes': [20, 40, 160, None]
    }
]
clf_rf = GridSearchCV(pipe_rf, search_space_rf, scoring=['accuracy', 'f1_micro', 'roc_auc_ovr'],
                      refit=False, verbose=3, n_jobs=-1, pre_dispatch='2*n_jobs')

# Initialize the KNN's Grid Search
pipe_knn = Pipeline([('std', StandardScaler()),
                     ('classifier', KNeighborsClassifier())])
n_neighbors = np.arange(1, 2000, 70)
distance = ['uniform', 'distance']
search_space_knn = [
    {
        'classifier': [KNeighborsClassifier(n_jobs=-1)],
        'classifier__weights': distance,
        'classifier__n_neighbors': n_neighbors
    }
]
clf_knn = GridSearchCV(pipe_knn, search_space_knn, scoring=['accuracy', 'f1_micro', 'roc_auc_ovr'],
                       refit=False, verbose=3, n_jobs=-1, pre_dispatch='2*n_jobs')

dataset_lists_long = {'adult': (encoded_adult, '>50K'),
                      'cape': (encoded_cape, 'GPA_Received')}

dataset_lists_short = {'letter.p1': (encoded_letter_p1, 'letter'),
                       'letter.p2': (encoded_letter_p2, 'letter'), 'cover': (encoded_cover, 'Cover_Type')}


def batch_grid_search(datasets, computer):
    clf_list = [clf_logistical, clf_rf, clf_knn, clf_svm, clf_mlp]
    scaler_all = StandardScaler()

    num_trails = 5
    sample_size = 5000
    best_model_param_list_batch = []
    grid_search_result_batch = []
    total_time = 0
    name = ''

    for name, dataset in datasets.items():
        start_data = time.time()
        dataset_df = dataset[0]
        dataset_y_col = dataset[1]
        for _ in range(num_trails):
            start = time.time()
            X_train, X_test, y_train, y_test = test_train_split_testing_size(dataset_df, sample_size, dataset_y_col)
            for clf in clf_list:
                clf_clone = clone(clf)
                runtime_start = time.time()
                best_model = clf_clone.fit(X_train, y_train)
                grid_search_result_batch.append(best_model.cv_results_)
                param = best_model.cv_results_['params'][np.argmin(best_model.cv_results_['rank_test_accuracy'])]
                best_model_refit = creat_best_model(param)  # create a new clf using the optimal params
                print("Currently work on " + name + " dataset's test")
                # transform the training data to work accordingly to the
                # Model we train.
                scaler = clone(scaler_all)
                best_model_refit.fit(scaler.fit_transform(X_train), y_train)
                metric = eval_model(best_model_refit, scaler.fit_transform(X_test), y_test)
                runtime_end = time.time()
                append_result(param, metric)
                param['dataset'] = name
                param['runtime'] = runtime_end - runtime_start
                best_model_param_list_batch.append(param)
            end = time.time()
            print(f"Searching, fitting {name}, it took {end - start}")
            total_time += end - start

        end_data = time.time()
        data_time = end_data - start_data
        message = f"\n\n PARTIALLY COMPLETED \n" \
                  f" {name} dataset completed. \n\n {name} took {data_time} seconds (" \
                  f"which is {int(data_time / 60)} minutes) to finish. \n\n" \
                  f"FYI: the training data sequence is {datasets}"
        send_email_after_finished(message)

    # Dump the data for future analysis
    df = parse_to_dataframe(best_model_param_list_batch)
    joblib.dump(df, f"out/{computer}_batch{name}.df")
    joblib.dump(best_model_param_list_batch, f"out/{computer}best_model_param{name}.pkl")
    joblib.dump(grid_search_result_batch, f"out/{computer}grid_search_result_batch{name}.pkl")

    print("total time to execute one trail one dataset is")
    print(int(total_time))
    message = str(best_model_param_list_batch) + f" \n \n {computer} " \
                                                 f"total time to complete is {int(total_time)} seconds, \n" \
                                                 f" which is {int(total_time / 60)} minutes" \
                                                 f" \n \n \n {grid_search_result_batch}"
    send_email_after_finished(message)
