import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from TelescopeML.DataMaster import *
from TelescopeML.DeepTrainer import *
from TelescopeML.Predictor import *
from TelescopeML.IO_utils import load_or_dump_trained_model_CNN
from TelescopeML.StatVisAnalyzer import *


class UnsupervisedMethods:
    def __init__(self, dataframe):
        self.data = dataframe
        self.data_processor = DataProcessor(self.data)        

    # def process_data(self):
    #     self.data_processor.split_train_validation_test(test_size=0.1, val_size=0.1, random_state_=42)
    #     self.data_processor.standardize_X_row_wise()

    def kmeans_clustering(self, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters)
        self.data['cluster'] = kmeans.fit_predict(self.data)
        return self.data


    