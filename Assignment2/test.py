# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler
from train import glucoseFeatures
import pickle
from sklearn.decomposition import PCA
import pickle_compat
pickle_compat.patch()


with open("RF_Model.pkl", 'rb') as file:
        GPC_Model = pickle.load(file) 
        test_df = pd.read_csv('test.csv', header=None)
    

cgm_features=glucoseFeatures(test_df)
ss_fit = StandardScaler().fit_transform(cgm_features)
    
pca = PCA(n_components=5)
pca_fit=pca.fit_transform(ss_fit)
    
predictions = GPC_Model.predict(pca_fit)
pd.DataFrame(predictions).to_csv("Results.csv", header=None, index=False)
