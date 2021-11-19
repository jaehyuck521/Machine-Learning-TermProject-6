import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from check_dataset import get_string_col
from check_dataset import get_numeric_col

import time
import datetime

import warnings

warnings.filterwarnings(action='ignore')

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


def encoding_column(encoder, X_scaled, X):
    for str_col in get_string_col(X):
        X_encoded = encoder.fit_transform(X[str_col].to_numpy().reshape(-1, 1))
        X_encoded = np.concatenate((X_scaled, X_encoded.reshape(X_scaled.shape[0], -1)), axis=1)

    return X_encoded


def autoMl(X_modeling, y_modeling, scalers, encoders, models, params_dict):
    best_accuracy = {}
    train_param_result = params_dict.copy()
    # find the best parameter by using grid search
    for scaler_key, scaler in scalers.items():
        print(f'--------scaler: {scaler_key}--------')
        X_scaled = scaler.fit_transform(X_modeling[get_numeric_col(X_modeling)])

        for encoder_key, encoder in encoders.items():
            print(f'------encoder: {encoder_key}------')
            X_encoded = encoding_column(encoder, X_scaled, X_modeling)

            # train test split
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_modeling, test_size=0.2)

            for model_key, model in models.items():
                print(f'----model: {model_key}----')

                start_time = time.time()  # for check running time

                # grid search
                grid = GridSearchCV(estimator=model, param_grid=params_dict[model_key], cv=5, verbose=2)
                grid.fit(X_train, y_train)
                print(f'params: {grid.best_params_}')
                best_model = grid.best_estimator_
                predict = best_model.predict(X_test)
                score = accuracy_score(y_test, predict)

                train_score = grid.cv_results_['mean_test_score']
                train_params = grid.cv_results_['params']

                print(f'-----------------\n{train_score}')
                print(f'-----------------\n{train_params}')

                flag = False
                for s, ps in zip(train_score, train_params):
                    if flag is False:
                        pre_s = s
                        pre_ps = ps
                        flag = True
                    else:
                        for key in pre_ps.keys():
                            if pre_ps[key] != ps[key]:
                                if pre_s > s:
                                    train_param_result[model_key][key] = -1
                                elif pre_s < s:
                                    train_param_result[model_key][key] = 1
                                else:
                                    train_param_result[model_key][key] = 0

                target_dict = {'model': model_key, 'score': score, 'estimator': best_model, 'scaler': scaler_key,
                               'encoder': encoder_key, 'param': grid.best_params_}

                if model_key not in best_accuracy.keys():
                    best_accuracy[model_key] = {'score': -1}
                # if model_key not in best_accuracy.keys():
                #     best_accuracy[model_key] = []

                elif best_accuracy[model_key]['score'] < target_dict['score']:
                    best_accuracy[model_key] = target_dict
                # best_accuracy[model_key].append(target_dict)

                print(f'score: {score}')
                end_time = time.time()  # for check running time
                print(f'running time: {end_time - start_time}  cur_time: {datetime.datetime.now()}', end='\n\n')

    return train_param_result, best_accuracy


def validate(X, y, scalers, encoders, best_param_dict):
    print(f"------------------------------------------")
    print(f"validate:")

    for model_key, param_info in best_param_dict.items():
        scaler = scalers[param_info['scaler']]
        encoder = encoders[param_info['encoder']]
        best_param = param_info['param']
        model = param_info['estimator']

        print(
            f"model: {model_key}\tscaler: {param_info['scaler']}\tencoder: {param_info['encoder']}\tparam: {best_param}")

        X_scaled = scaler.fit_transform(X[get_numeric_col(X)])
        X_encoded = encoding_column(encoder, X_scaled, X)

        predict = model.predict(X_encoded)

        score = accuracy_score(y, predict)
        confuse_mat = confusion_matrix(y, predict)
        report = classification_report(y, predict, digits=5)

        print(f'score: {score}')
        print(f'confuse matrix:\n{confuse_mat}')
        print(f'{report}')


def getBestParam(best_accuracy):
    best_param = {}
    for model_name, result in best_accuracy.items():
        best_param[model_name] = {}
        for k, v in result.items():
            if k in ['estimator', 'scaler', 'encoder', 'param']:
                best_param[model_name][k] = v

    return best_param


def getVotingModel(models, voting_list, best_accuracy):
    voting_estimator_list = []
    voting_estimator_param_list = []

    for vl in voting_list:
        estimator = models[vl]
        estimator.set_params(**best_accuracy[vl]['param'])
        temp_param = best_accuracy[vl]['param']
        temp_param["model"] = vl
        voting_estimator_param_list.append(temp_param)
        voting_estimator_list.append((vl, estimator))

    Voting = VotingClassifier(estimators=voting_estimator_list)

    return Voting


# function for find the best accuracy given datas, params
def find_best(X, y, scalers, encoders, models, params_dict, voting_list):
    # make validation set
    X_modeling, X_val, y_modeling, y_val = train_test_split(X, y, test_size=0.2)

    train_param, best_accuracy = autoMl(X_modeling, y_modeling, scalers, encoders, models, params_dict)

    # make voting classifier's parameter
    voting_param = {}
    voting_param["Voting"] = {}
    voting_param["Voting"]["voting"] = ["hard", "soft"]

    Voting = getVotingModel(models, voting_list, best_accuracy)

    voting_model = {"Voting": Voting}

    voting_train_param, voting_accuracy = autoMl(X_modeling, y_modeling, scalers, encoders, voting_model, voting_param)

    best_accuracy.update(voting_accuracy)

    train_param.update(voting_train_param)

    best_param = getBestParam(best_accuracy)

    # validate(X_val, y_val, scalers, encoders, best_param)

    return train_param, best_accuracy


# function for set hyper parameters and run find_best
def train():
    df = pd.read_csv(r'./dataset/afterPreprocessing.csv')

    df = df.iloc[:20, :]

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

    # 3. Model : XGB, gradientBoosting, (Decision tree(entropy), Decision tree(Gini), GaussianNB, knn, random forest)

    XGB = xgb.XGBClassifier(booster='gbtree', nthread=4, eval_metric='mlogloss')
    decision_tree_gini = DecisionTreeClassifier(criterion='gini')
    decision_tree_entropy = DecisionTreeClassifier(criterion='entropy')
    knn = KNeighborsClassifier()
    gboost = GradientBoostingClassifier()
    rf = RandomForestClassifier()
    NB = GaussianNB()

    # save scalers and models and hyper parameters in dictionary

    scalers = {"standard scaler": standard, "minMax scaler": minMax, "robust scaler": robust}

    encoders = {"one-hot encoder": oneHot_encoder, "ordinal encoder": ordinal_encoder}

    models = {'xgb': XGB, 'gradient_boosting': gboost, 'random_forest': rf, 'knn': knn,
              'decision_tree_gini': decision_tree_gini, 'decision_tree_entropy': decision_tree_entropy,
              'NaiveBayes': NB}

    lr = 1.1
    # params_lr_dict = {
    #     'xgb': {'min_child_weight': 2, 'max_depth': 2,
    #             'gamma': 1.1},
    #
    #     'gradient_boosting': {'learning_rate': 1.1, 'n_estimators': 2,
    #                           'max_depth': 2},
    #
    #     'knn': {'n_neightbors': 2},
    #
    #     'random_forest': {'n_estimators': 2},
    #
    #     "decision_tree_entropy": {"max_depth": 2,
    #                               "min_samples_split": 2,
    #                               "min_samples_leaf": 2},
    #     "decision_tree_gini": {"max_depth": 1.1,
    #                            "min_samples_split": 2,
    #                            "min_samples_leaf": 2},
    #
    #     'NaiveBayes': {'var_smoothing': 10}
    # }

    params_dict_origin = {
        'xgb': {'min_child_weight': [1, 64], 'max_depth': [2, 64],
                'gamma': [0.1, 1]},

        'gradient_boosting': {'learning_rate': [0.1, 1], 'n_estimators': [2, 512],
                              'max_depth': [1, 16]},

        'knn': {'n_neightbors': [1, 16]},

        'random_forest': {'n_estimators': [1, 64]},

        "decision_tree_entropy": {"max_depth": [2, 64],
                                  "min_samples_split": [0.1, 1],
                                  "min_samples_leaf": [2, 16]},

        "decision_tree_gini": {"max_depth": [2, 64],
                               "min_samples_split": [0.1, 1],
                               "min_samples_leaf": [2, 16]},

        'NaiveBayes': {'var_smoothing': [10e-9, 0.1]}
    }

    voting_list = ["decision_tree_entropy", "decision_tree_gini", "NaiveBayes", 'knn', 'random_forest']

    for epoch in range(1, 100):
        train_param, result_dict = find_best(X=X, y=y, scalers=scalers, encoders=encoders, models=models,
                                             params_dict=params_dict_origin,
                                             voting_list=voting_list)

        for model_key, params in params_dict_origin:
            for param_key in params.keys():
                if train_param[model_key][param_key] == 1:
                    params_dict_origin[param_key] *= lr
                elif train_param[model_key][param_key] == -1:
                    params_dict_origin[param_key] /= lr


    return result_dict


if __name__ == "__main__":
    result = train()

