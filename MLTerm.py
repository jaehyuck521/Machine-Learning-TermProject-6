import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN,SpectralClustering
from sklearn.preprocessing import Normalizer,StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.metrics import silhouette_score,confusion_matrix
import warnings
warnings.filterwarnings(action='ignore')

df=pd.read_csv('afterPreprocessing.csv')
print(df.info())
priceDivide=df['price']
df.loc[df['price']<=10000,'class']='A'
df.loc[(df['price']>10000) & (df['price']<=20000),'class']='B'
df.loc[(df['price']>20000) & (df['price']<=30000),'class']='C'
df.loc[(df['price']>30000) & (df['price']<=50000),'class']='D'
df.loc[(df['price']>50000) ,'class']='E'
df.drop('price',axis=1,inplace=True)
encoder=OrdinalEncoder()
scaler=StandardScaler()

print(df.info())

# df_needEncoding=df[['manufacturer','model','condition','cylinders','fuel','title_status','transmission','VIN','drive','size','type', 'paint_color','state','class']]
df_needEncoding=df[['region','manufacturer','condition','cylinders','fuel','title_status','transmission','drive','type', 'paint_color','state','class']]
df_notEncoding=df.drop(['region','manufacturer','condition','cylinders','fuel','title_status','transmission','drive','type', 'paint_color','state','class'],axis=1)
df_notEncoding=df_notEncoding.drop('Unnamed: 0', axis=1)
df_notEncoding=pd.DataFrame(scaler.fit_transform(df_notEncoding),columns=df_notEncoding.columns)
df_encoding=pd.DataFrame(encoder.fit_transform(df_needEncoding), columns=df_needEncoding.columns)
df_heatmap=pd.concat([df_encoding,df_notEncoding],axis=1)
# print(df_heatmap.head(5))
# print(df_heatmap.head(5))
# print("sample covariance matrix")
# covMatrix=np.cov(df_heatmap,bias=True)# sample data
# print(covMatrix)
# sns.heatmap(covMatrix,annot=True, fmt='g')
# plt.show()
# heatmap by seaborn
# df['age'] = df.posting_date.dt.year.astype(int) - df.year.astype(int)
# ax = sns.heatmap(df_heatmap)
# plt.title('Heatmap with seaborn', fontsize=20)
# plt.show()

# sns.heatmap(df_heatmap, cmap='RdYlGn_r')
# plt.title('Heatmap with changing color', fontsize=20)
# plt.show()
# colormap = plt.cm.PuBu
# plt.figure(figsize=(20, 20))
# plt.title("Used car list heatmap with correlation", y = 1.05, size = 20)
# sns.heatmap(df_heatmap.astype(float).corr(), linewidths = 1.0, vmax = 1.0, square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 5})
# plt.show()

# print(df['class'])
# print(df[df['class']=='E'].count())
# ratio=[14258,9595, 4721, 3869, 608]
# plt.pie(ratio, labels=['A','B','C','D','E'],autopct='%.1f%%',explode=[0, 0.10, 0, 0.10,0])
# plt.show()
# print(df['cylinders'].unique())
# print(df[df['cylinders']=='12 cylinders'].count())

# ratio=[11103,10870, 10124,272,548,47,82,5]
# plt.pie(ratio, labels=df['cylinders'].unique(),autopct='%.1f%%',explode=[0,0,0,0,0,0.1,0.1,0.1])
# plt.show()

x=df_heatmap.drop('class',axis=1)
y=df_heatmap['class'].values.reshape(-1,1)
def Xgboost(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    xgb_reg =XGBClassifier()
    xgb_reg.fit(x_train, y_train)
    print("\t------- XGBoost Regression -------")
    score=xgb_reg.score(x_test,y_test)
    return score

# print(Xgboost(x,y))

def k_means(X):
    pca=PCA(n_components=2)
    n_clusters=[2,3,4,5,6,7]
    n_init=[8,9,10,11,12]
    distortions = []
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(2, 9), distortions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.show()
    list={}
    for i in range(0,len(n_clusters)):
        plt.figure(figsize=(16, 4))
        plt.rc("font",size=5)
        # SUBPLOT POSITION
        position = 1
        for j in range(0, len(n_init)):
                    X=pca.fit_transform(X)
                    model = KMeans(random_state=0, n_clusters=n_clusters[i], init='k-means++', max_iter=300,algorithm='full',n_init=n_init[j])
                    model.fit(X)
                    label=model.labels_
                    cluster_id=pd.DataFrame(label)
                    kx=pd.DataFrame(X)
                    k1=pd.concat([kx,cluster_id],axis=1)
                    k1.columns=['p1','p2',"cluster"]
                    labeled=k1.groupby("cluster")
                    plt.subplot(1, 5, position)
                    score = silhouette_score(X, label, metric="euclidean")

                    plt.title("N_init={algo} Score={score}".format(algo=n_init[j], score=round(score, 3)))
                    for cluster, pos in labeled:
                        if cluster == -1:
                            # NOISE WITH COLOR BLACK
                            plt.plot(pos.p1, pos.p2, marker='o', linestyle='', color='black')
                        else:
                            plt.plot(pos.p1, pos.p2, marker='o', linestyle='')
                    position += 1
        print("N_components={n} The purity score: ".format(n=n_clusters[i]),purity_print(X, label, n_clusters[i]))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.suptitle("K-Means: N_CLUSTERS={0}".format(n_clusters[i]))
        plt.show()

def mean_Shift(X):
    min_bin_freq=[1,3,5,7,9,11]
    pca = PCA(n_components=2)
    sampleList=[3000,5000,10000,30000]
    for i in range(0,len(sampleList)):
        bandwidth=estimate_bandwidth(X,n_samples=sampleList[i])
        plt.figure(figsize=(16, 4))
        plt.rc("font", size=5)
        position = 1
        for j in range(0, len(min_bin_freq)):
            model=MeanShift(bandwidth=bandwidth, cluster_all=True,max_iter=500,min_bin_freq=min_bin_freq[j])
            X=pca.fit_transform(X)
            model.fit(X)
            labels=model.labels_
            cluster_id = pd.DataFrame(labels)
            kx = pd.DataFrame(X)
            k1 = pd.concat([kx, cluster_id], axis=1)
            k1.columns = ['p1', 'p2', "cluster"]
            labeled = k1.groupby("cluster")
            plt.subplot(1, 6, position)
            # score = silhouette_score(X, labels, metric="euclidean")
            plt.title("Min_bin_freq={maxiter}".format(maxiter=min_bin_freq[j]))
            for cluster, pos in labeled:
                if cluster == -1:
                    # NOISE WITH COLOR BLACK
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='', color='black')
                else:
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='')
            position += 1
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.suptitle("MeansShift: N_samples={0}".format(sampleList[i]))
        plt.show()

# k_means(x)
def gaussian_Mixture(X):
    n_components=[2,3,4,5,6,7]
    cov_type=['full','tied','diag','spherical']
    pca=PCA(n_components=2)
    for i in range(0, len(n_components)):
        plt.figure(figsize=(16, 4))
        plt.rc("font", size=5)
        position = 1
        for j in range(0, len(cov_type)):
            model = GaussianMixture(n_components=n_components[i], covariance_type=cov_type[j], max_iter=500, init_params='kmeans',warm_start=False)
            X = pca.fit_transform(X)
            model.fit(X)
            gaussian_label=model.predict(X)
            # print(gaussian_label)
            cluster_id = pd.DataFrame(gaussian_label)
            kx = pd.DataFrame(X)
            k1 = pd.concat([kx, cluster_id], axis=1)
            k1.columns = ['p1', 'p2', "cluster"]
            labeled = k1.groupby("cluster")
            plt.subplot(1, 4, position)
            score = silhouette_score(X, gaussian_label, metric="euclidean")
            plt.title("covariance type={maxiter} Score={score}".format(maxiter=cov_type[j], score=round(score, 3)))
            for cluster, pos in labeled:
                if cluster == -1:
                    # NOISE WITH COLOR BLACK
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='', color='black')
                else:
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='')
            position += 1
        print("N_components={n} The purity score: ".format(n=n_components[i]), purity_print(X, gaussian_label, n_components[i]))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.suptitle("Gaussian Mixture : N_components={0}".format(n_components[i]))
        plt.show()

def db_scan(X):
    eps=[0.05,0.1, 0.5, 1, 2]
    min_sample=[5,10,15]
    pca = PCA(n_components=2)
    X = pd.DataFrame(pca.fit_transform(x))
    for i in range(0,len(min_sample)):
        plt.figure(figsize=(16, 4))
        plt.rc("font", size=5)
        position = 1
        for j in range(0, len(eps)):
            model=DBSCAN(eps=eps[j], min_samples=min_sample[i])
            labels=model.fit_predict(X)
            cluster_id = pd.DataFrame(labels)
            kx = pd.DataFrame(X)
            k1 = pd.concat([kx, cluster_id], axis=1)
            k1.columns = ['p1', 'p2', "cluster"]
            labeled = k1.groupby("cluster")
            plt.subplot(1, 6, position)
            score = silhouette_score(X, labels, metric="euclidean")
            plt.title("eps={maxiter} score: {sc}".format(maxiter=eps[j],sc=round(score,3)))
            for cluster, pos in labeled:
                if cluster == -1:
                    # NOISE WITH COLOR BLACK
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='', color='black')
                else:
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='')
            position += 1
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.suptitle("DBSCAN: MIN_SAMPLES={0}".format(min_sample[i]))
        plt.show()

def purity_score(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)  # compute confusion matrix
    return np.sum(np.amax(cf_matrix, axis=0)) / np.sum(cf_matrix)

def purity_print(x,y, quantile):
    labels_price = list(map(str, np.arange(0, quantile)))
    labeled_price = pd.cut(priceDivide, quantile, labels=labels_price,include_lowest=True)
    x=pd.DataFrame(x)
    new_x = pd.concat([x, labeled_price], axis=1)
    new_x['price'] = pd.to_numeric(new_x['price'])
    score=purity_score(new_x['price'], y)
    return score
def spectral_clus(X):
    pca = PCA(n_components=2)
    n_cluster=[2,3,4,5,6,7]
    affinity=['rbf','nearest_neighbors']
    for i in range(0, len(n_cluster)):
        plt.figure(figsize=(16, 4))
        plt.rc("font", size=5)
        position = 1
        for j in range(0, len(affinity)):
            model = SpectralClustering(n_clusters=n_cluster[i], affinity=affinity[j])
            X = pca.fit_transform(X)
            model.fit(X)
            gaussian_label=model.predict(X)
            # print(gaussian_label)
            cluster_id = pd.DataFrame(gaussian_label)
            kx = pd.DataFrame(X)
            k1 = pd.concat([kx, cluster_id], axis=1)
            k1.columns = ['p1', 'p2', "cluster"]
            labeled = k1.groupby("cluster")
            plt.subplot(1, 2, position)
            score = silhouette_score(X, gaussian_label, metric="euclidean")
            plt.title("Affinity = {maxiter} Score={score}".format(maxiter=affinity[j], score=round(score, 3)))
            for cluster, pos in labeled:
                if cluster == -1:
                    # NOISE WITH COLOR BLACK
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='', color='black')
                else:
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='')
            position += 1
        print("N_clusters={n} The purity score: ".format(n=n_cluster[i]), purity_print(X, gaussian_label, n_cluster[i]))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.suptitle("Spectral clustering : N_components={0}".format(n_cluster[i]))
        plt.show()


# k_means(x)
print(x.columns)
# db_scan(x)
# k_means(x)
# mean_Shift(x)
# db_scan(x)
# gaussian_Mixture(x)
# purity score code
# spectral_clus(x)



# elbow method visualization
def visualize_silhouette_layer(data, param_init='random', param_n_init=10, param_max_iter=300):
    clusters_range = range(2,8)
    results = []
    for i in clusters_range:
        clusterer = KMeans(n_clusters=i, init=param_init, n_init=param_n_init, max_iter=param_max_iter, random_state=0)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        results.append([i, silhouette_avg])

    result = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
    pivot_km = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

    plt.figure()
    sns.heatmap(pivot_km, annot=True, linewidths=.5, fmt='.3f', cmap=sns.cm._rocket_lut)
    plt.tight_layout()
    plt.show()

# visualize_silhouette_layer(x)



