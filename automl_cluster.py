import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

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


def cv_silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    score = silhouette_score(X, labels, metric='euclidean')
    return score



# function for find the best accuracy given datas, params
def find_best(X, scalers, encoders, models, params_dict=None):
    # save the best accuracy each models
    global best_params
    global score
    score = -1
    best_accuracy = {}

    # find the best parameter by using grid search

    for scaler_key, scaler in scalers.items():
        print(f'--------scaler: {scaler_key}--------')
        X_scaled = scaler.fit_transform(X[get_numeric_col(X)])

        for encoder_key, encoder in encoders.items():
            print(f'------encoder: {encoder_key}------')
            X_encoded = X_scaled.copy()
            for str_col in get_string_col(X):
                X_encoded = encoder.fit_transform(X[str_col].to_numpy().reshape(-1, 1))
                X_encoded = np.concatenate((X_scaled, X_encoded.reshape(X_scaled.shape[0], -1)), axis=1)

            for model_key, model in models.items():
                print(f'----model: {model_key}----')
                start_time = time.time()  # for check running time
                if model_key=='meanshift':
                    param_list = list(params_dict[model_key].keys())
                    meanshift_params = params_dict[model_key]
                    for p1 in meanshift_params[param_list[0]]:
                        for p2 in meanshift_params[param_list[1]]:
                            temp_params = {param_list[0]: p1, param_list[1]: p2}
                            bandwidth=estimate_bandwidth(X_encoded,n_samples=p2)
                            meanshift_model=MeanShift(bandwidth=bandwidth, cluster_all=True,max_iter=500,min_bin_freq=p1)
                            meanshift_model.fit(X_encoded)
                            labels = meanshift_model.labels_
                            temp_score = silhouette_score(X_encoded, labels, metric='euclidean')
                            if temp_score > score:
                                score = temp_score
                                best_params = temp_params

                else:
                    cv = [(slice(None), slice(None))]
                    grid = GridSearchCV(estimator=model, param_grid=params_dict[model_key], scoring=cv_silhouette_scorer,
                                        cv=cv)
                    grid.fit(X_encoded)
                    best_params = grid.best_params_
                    score = grid.best_score_
                    print(f'params: {best_params}')
                # save the 3 highest accuracy and parameters each models
                save_len = 3
                save_len -= 1
                flag = False

                target_dict = {'score': score, 'model': model_key, 'scaler': scaler_key,
                               'encoder': encoder_key, 'param': best_params}
                # save accuracy if best_accuracy has less than save_len items
                if model_key not in best_accuracy.keys():
                    best_accuracy[model_key] = []
                if len(best_accuracy[model_key]) <= save_len:
                    best_accuracy[model_key].append(target_dict)
                    best_accuracy[model_key].sort(key=lambda x: x['score'], reverse=True)
                # insert accuracy for descending
                elif best_accuracy[model_key][-1]['score'] < score:
                    for i in range(1, save_len):
                        if best_accuracy[model_key][save_len - 1 - i]['score'] > score:
                            best_accuracy[model_key].insert(save_len - i, target_dict)
                            best_accuracy[model_key].pop()
                            flag = True
                            break
                    if flag is False:
                        best_accuracy[model_key].insert(0, target_dict)
                        best_accuracy[model_key].pop()

                print(f'score: {score}', end='')
                end_time = time.time()  # for check running time
                print(f'   running time: {end_time - start_time}  cur_time: {datetime.datetime.now()}', end='\n\n')

    print(f'------train result------')
    displayResultDict(best_accuracy)

    return best_accuracy


def get_numeric_col(df):
    numeric_col_list = []

    for col_name in df.columns:
        if is_numeric_dtype(df[col_name].dtypes):
            numeric_col_list.append(col_name)

    return numeric_col_list


def get_string_col(df):
    string_col_list = []

    for col_name in df.columns:
        if is_string_dtype(df[col_name].dtypes):
            string_col_list.append(col_name)

    return string_col_list


def outlier_iqr(df):
    numeric_col_list = get_numeric_col(df)

    for col_name in numeric_col_list:
        q1, q3 = np.percentile(df[col_name], [25, 75])

        iqr = q3 - q1

        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)

        df = df[upper_bound > df[col_name]]
        df = df[df[col_name] > lower_bound]

    return df


def preprocessing():
    df = pd.read_csv('afterPreprocessing.csv')
    df_prep=df.drop('price',axis=1)
    return df_prep


# function for set hyper parameters and run find_best
def train():
    X = preprocessing()

    # 1. Scaler : Standard, MinMax, Robust

    standard = StandardScaler()
    minMax = MinMaxScaler()
    robust = RobustScaler()

    # 2. Encoder : Label, One-Hot

    label_encoder = LabelEncoder()
    ordinal_encoder=OrdinalEncoder()

    # 3. Model : Decision tree(entropy), Decision tree(Gini), Logistic regression, SVM

    kmeans = KMeans()
    dbscan = DBSCAN()
    em = GaussianMixture()
    meanshift = MeanShift()

    # save scalers and models and hyper parameters in dictionary

    scalers = {"standard scaler": standard, "minMax scaler": minMax, "robust scaler": robust}

    encoders = {"ordinal encoder": ordinal_encoder, "label encoder": label_encoder}

    models = {"kmeans": kmeans, "dbscan": dbscan,
              "em": em, 'meanshift':meanshift}

    # params_dict = {"kmeans": {"n_clusters": range(2, 12), "tol": [1e-6, 1e-4, 1e-2, 1]},
    #                "dbscan": {"eps": [0.2, 0.5, 0.8], "min_samples": [3, 5, 7, 9]},
    #                "em": {"n_components": [1, 2, 3], "tol": [1e-5, 1e-3, 1e-1, 10]},
    #                "spectral": {"n_clusters": range(2, 12), "gamma": [1, 2]},
    #                "clarans_model": {"number_clusters": range(3, 4), "numlocal": [4, 6, 8],
    #                                  "maxneighbor": [2, 4, 6]}
    #                }

    params_dict = {"kmeans": {"n_clusters": range(2, 8), "n_init": [8,9,10,11,12]},
                   "dbscan": {"eps": [0.05,0.1, 0.5, 1, 2], "min_samples": [5,30,100]},
                   "em": {"n_components": range(2,8), "covariance_type": ['full','tied','diag','spherical']},
                   "meanshift": {'min_bin_freq':[1,3,5,7,9,11], "n_samples": [3000,5000,10000,30000]}
                   }

    find_best(X, scalers, encoders, models, params_dict)


# function for display result_dict
def displayResultDict(result_dict):
    print(result_dict)
    for model_name, result_list in result_dict.items():
        print(model_name)
        for result in result_list:
            print(result)


if __name__ == "__main__":
    train()
