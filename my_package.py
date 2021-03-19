import pandas as pd
import numpy as np
import smtplib
import ssl
import joblib

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def clean_course_num(code):
    """
    Helper function to clean the cape stats that divide the course
    in to upper division and lower division

    :param code: the course code
    :return: 1 for upper, -1 for lower
    """
    code = str(code)
    while code[-1].isalpha():
        code = code[:-1]
    code = int(code)
    if code >= 100:
        return 1
    return -1


def creat_best_model(model):
    """
    create the best model using based on the grid search.
    This method will parse out the parameter and the classifier of the best parameter

    :param model: dictionary containing classifier
    :return: the classifier with the optimal parameter
    """
    out = {}
    assert 'classifier' in model
    for param, value in model.items():
        param = param[12:]
        if len(param) == 0:  # that means that is a classifier
            if isinstance(value, SVC):
                out['cache_size'] = 16000
            clf = value
        else:
            out[param] = value  # that means that is a param
    clf.set_params(**out)
    return clf


def eval_model(model, X_test, y_test):
    """
    return the performance score of three different error metric
    the average between accurary, F1 score, and AUC
    """
    performance = np.array([model.score(X_test, y_test),
                            roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
                            f1_score(y_test, model.predict(X_test))])
    return performance, performance.mean()


def append_result(param, metric):
    """
    Deep copy the metric and append to the param

    :param param: the dictionary we want to append
    :param metric: the metrics that has performance
    :return: the updated param
    """
    param['acc'] = metric[0][0]
    param['roc_auc'] = metric[0][1]
    param['f1_score'] = metric[0][2]
    param['performance'] = metric[1]
    return param


def parse_to_dataframe(models):
    """
    un-flatten the given nested dictionary for further model analysis purposes

    :param models: Nested Dictionary with the same keys
    :return: dataframe that has been un-flattened
    """
    keys = set()
    for dictionary in models:
        keys.update(list(dictionary.keys()))
    nested = dict()
    for key in keys:
        nested[key] = []
    for key in keys:
        for model in models:
            if key in model:
                nested[key].append(model[key])
            else:
                nested[key].append(None)
    return pd.DataFrame(nested)


def test_train_split_testing_size(df, training_size, y_name):
    """
    helper function to split the training set and testing set by a
    specific number of testing size

    :param df: the dataframe of encoded one
    :param training_size: the amount of observation we want to test
    :param y_name: the column name from that dataframe that is the label
    :return: X_train, X_test, y_train, y_test
    """
    assert y_name in df.columns

    X = df.drop(columns=y_name).to_numpy()
    y = df[y_name].values
    if df.shape[1] == 186:
        pca = PCA(n_components=70, svd_solver='full')
        X = pca.fit_transform(X)
    return train_test_split(X, y, train_size=training_size)


def send_email_after_finished(content):
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "yyj20010417@gmail.com"
    receiver_email = "yuy004@ucsd.edu"
    password = "Radiation657"
    message = f"""\
Subject: Your training set is Completed

Please navigate back to the server to check the updated data.

Good luck on your final. \n \n
{content}
    """

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
