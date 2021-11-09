import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# 3. Model : XGB, SVM, (Decision tree(entropy), Decision tree(Gini), GaussianNB)
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from check_dataset import get_string_col
from check_dataset import get_numeric_col

import time
import datetime

import warnings

# warnings.filterwarnings(action='ignore')

# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)

import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


def makePriceToClass(df):
    df.loc[df['price'] <= 10000, 'class'] = 0
    df.loc[(df['price'] > 10000) & (df['price'] <= 20000), 'class'] = 1
    df.loc[(df['price'] > 20000) & (df['price'] <= 30000), 'class'] = 2
    df.loc[(df['price'] > 30000) & (df['price'] <= 50000), 'class'] = 3
    df.loc[(df['price'] > 50000), 'class'] = 4
    df.drop('price', axis=1)
    return df


# function for find the best accuracy given datas, params
def find_best(X, y, scalers, encoders, models, params_dict, voting_list):
    # save the best accuracy each models
    global best_params
    global score
    score = -1
    best_accuracy = {}

    # make validation set
    X_modeling, X_val, y_modeling, y_val = train_test_split(X, y, test_size=0.001)

    # find the best parameter by using grid search
    for scaler_key, scaler in scalers.items():
        print(f'--------scaler: {scaler_key}--------')
        X_scaled = scaler.fit_transform(X_modeling[get_numeric_col(X_modeling)])

        for encoder_key, encoder in encoders.items():
            print(f'------encoder: {encoder_key}------')
            X_encoded = X_scaled.copy()
            for str_col in get_string_col(X_modeling):
                X_encoded = encoder.fit_transform(X_modeling[str_col].to_numpy().reshape(-1, 1))
                X_encoded = np.concatenate((X_scaled, X_encoded.reshape(X_scaled.shape[0], -1)), axis=1)

            # train test split
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_modeling, test_size=0.2)

            for model_key, model in models.items():
                print(f'----model: {model_key}----')

                start_time = time.time()  # for check running time

                # grid search
                grid = GridSearchCV(estimator=model, param_grid=params_dict[model_key])
                grid.fit(X_train, y_train)
                print(f'params: {grid.best_params_}')
                best_model = grid.best_estimator_
                predict = best_model.predict(X_test)
                score = accuracy_score(y_test, predict)

                target_dict = {'score': score, 'model': model_key, 'scaler': scaler_key,
                               'encoder': encoder_key, 'param': grid.best_params_}

                if model_key not in best_accuracy.keys():
                    best_accuracy[model_key] = []
                best_accuracy[model_key].append(target_dict)

                print(f'score: {score}', end='')
                end_time = time.time()  # for check running time
                print(f'   running time: {end_time - start_time}  cur_time: {datetime.datetime.now()}', end='\n\n')

    # save the 3 highest accuracy and parameters each models
    for model_key, model in models.items():
        best_accuracy[model_key].sort(key=lambda x: x['score'], reverse=True)
        best_accuracy[model_key] = best_accuracy[model_key][:3]

    print(f'------train result------')
    displayResultDict(best_accuracy)

    voting_param = {}
    voting_param["Voting"] = {}
    voting_param["Voting"]["voting"] = ["hard", "soft"]

    voting_estimator_list = []
    voting_estimator_param_list = []

    for vl in voting_list:
        estimator = models[vl]
        estimator.set_params(**best_accuracy[vl][0]['param'])
        temp_param = best_accuracy[vl][0]['param']
        temp_param["model"] = vl
        voting_estimator_param_list.append(temp_param)
        voting_estimator_list.append((vl, estimator))

    Voting = VotingClassifier(estimators=voting_estimator_list)

    voting_model = {"Voting": Voting}

    voting_accuracy = {}

    # find the best parameter by using grid search
    for scaler_key, scaler in scalers.items():
        print(f'--------scaler: {scaler_key}--------')
        X_scaled = scaler.fit_transform(X_modeling[get_numeric_col(X_modeling)])

        for encoder_key, encoder in encoders.items():
            print(f'------encoder: {encoder_key}------')
            X_encoded = X_scaled.copy()
            for str_col in get_string_col(X_modeling):
                X_encoded = encoder.fit_transform(X_modeling[str_col].to_numpy().reshape(-1, 1))
                X_encoded = np.concatenate((X_scaled, X_encoded.reshape(X_scaled.shape[0], -1)), axis=1)

            # train test split
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_modeling, test_size=0.2)

            for model_key, model in voting_model.items():
                print(f'----model: {model_key}----')

                start_time = time.time()  # for check running time

                # grid search
                grid = GridSearchCV(estimator=model, param_grid=voting_param[model_key])
                grid.fit(X_train, y_train)
                print(f'params: {grid.best_params_}')
                best_model = grid.best_estimator_
                predict = best_model.predict(X_test)
                score = accuracy_score(y_test, predict)

                target_dict = {'score': score, 'model': model_key, 'scaler': scaler_key,
                               'encoder': encoder_key, 'param': grid.best_params_}

                if model_key not in voting_accuracy.keys():
                    voting_accuracy[model_key] = []
                voting_accuracy[model_key].append(target_dict)

                print(f'score: {score}', end='')
                end_time = time.time()  # for check running time
                print(f'   running time: {end_time - start_time}  cur_time: {datetime.datetime.now()}', end='\n\n')

    # save the 3 highest accuracy and parameters voting models
    voting_accuracy["Voting"].sort(key=lambda x: x['score'], reverse=True)
    voting_accuracy["Voting"] = voting_accuracy["Voting"][:3]

    print(f'------train result------')
    print(f'voting estimator params: {voting_estimator_param_list}')
    displayResultDict(voting_accuracy)

    return best_accuracy


# function for set hyper parameters and run find_best
def train():
    df = pd.read_csv(r'./dataset/afterPreprocessing.csv')

    # df = df.iloc[:100, :]

    df_make_class = makePriceToClass(df.copy())

    target_col = 'class'

    X = df_make_class.drop(target_col, axis=1)
    y = df_make_class[target_col]

    # 1. Scaler : Standard, MinMax, Robust

    standard = StandardScaler()
    minMax = MinMaxScaler()
    robust = RobustScaler()

    # 2. Encoder : Label, One-Hot

    ordinal_encoder = OrdinalEncoder()
    oneHot_encoder = OneHotEncoder(sparse=False)

    # 3. Model : XGB, SVM, (Decision tree(entropy), Decision tree(Gini), GaussianNB)

    XGB = xgb.XGBClassifier(booster='gbtree', nthread=4, eval_metric='mlogloss')
    SVM = SVC()
    decision_tree_gini = DecisionTreeClassifier(criterion='gini')
    decision_tree_entropy = DecisionTreeClassifier(criterion='entropy')
    NB = GaussianNB()

    # save scalers and models and hyper parameters in dictionary

    scalers = {"standard scaler": standard, "minMax scaler": minMax, "robust scaler": robust}

    encoders = {"one-hot encoder": oneHot_encoder, "ordinal encoder": ordinal_encoder}

    # models = {'xgb': XGB, 'svm': SVM, 'decision_tree_gini': decision_tree_gini, 'decision_tree_entropy': decision_tree_entropy, 'NaiveBayes': NB}
    models = {'svm': SVM, 'decision_tree_gini': decision_tree_gini,
              'decision_tree_entropy': decision_tree_entropy, 'NaiveBayes': NB}

    params_dict = {'xgb': {'min_child_weight': [2**r for r in range(0, 5)], 'max_depth': [2**r for r in range(1, 6)], 'gamma': [r/10 for r in range(0, 3)]},

                   'svm': {'C': [r/1000 for r in range(1, 3002, 400)], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                           'degree': [r/10 for r in range(1, 60, 10)], 'gamma': ['scale', 'auto'], 'tol': [r/1000 for r in range(1, 2000, 500)]
                           },

                   "decision_tree_entropy": {"max_depth": [r for r in range(3, 11, 1)],
                                            "min_samples_split": [r / 101 for r in range(2, 50, 5)] + [2, 4, 8, 16],
                                            "min_samples_leaf": [r for r in range(5, 10, 1)]},
                   "decision_tree_gini": {"max_depth": [r for r in range(3, 11, 1)],
                                         "min_samples_split": [r / 101 for r in range(2, 50, 5)] + [2, 4, 8, 16],
                                         "min_samples_leaf": [r for r in range(5, 10, 1)]},

                   'NaiveBayes': {'var_smoothing': [10**r for r in range(-9, -1)]}
                   }

    # params_dict = {
    #     'xgb': {'min_child_weight': [2 ** r for r in range(0, 2)], 'max_depth': [2 ** r for r in range(1, 2)],
    #             'gamma': [r / 10 for r in range(0, 2)]},
    #
    #     'svm': {'C': [r / 1000 for r in range(1, 800, 400)], 'kernel': ['rbf', 'sigmoid'],
    #             'degree': [r / 10 for r in range(1, 30, 10)], 'gamma': ['scale', 'auto'],
    #             'tol': [r / 1000 for r in range(1, 1000, 500)]
    #             },
    #
    #     "decision_tree_entropy": {"max_depth": [r for r in range(3, 5, 1)],
    #                               "min_samples_split": [r / 101 for r in range(2, 10, 5)] + [2],
    #                               "min_samples_leaf": [r for r in range(5, 6, 1)]},
    #     "decision_tree_gini": {"max_depth": [r for r in range(3, 5, 1)],
    #                            "min_samples_split": [r / 101 for r in range(2, 10, 5)] + [2],
    #                            "min_samples_leaf": [r for r in range(5, 6, 1)]},
    #
    #     'NaiveBayes': {'var_smoothing': [10 ** r for r in range(-3, -1)]}
    #     }

    voting_list = ["decision_tree_entropy", "decision_tree_gini", "NaiveBayes"]

    find_best(X=X, y=y, scalers=scalers, encoders=encoders, models=models, params_dict=params_dict,
              voting_list=voting_list)


# function for display result_dict
def displayResultDict(result_dict):
    print(result_dict)
    for model_name, result_list in result_dict.items():
        print(model_name)
        for result in result_list:
            print(result)


if __name__ == "__main__":
    train()

