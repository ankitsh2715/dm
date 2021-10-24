import pandas as pd
import pickle
import pickle_compat
from train import glucoseFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pickle_compat.patch()

with open("RF_Model.pkl", "rb") as file:
    model = pickle.load(file)
    test_df = pd.read_csv("test.csv", header=None)

glucose_features = glucoseFeatures(test_df)
fit_SS = StandardScaler().fit_transform(glucose_features)

pca = PCA(n_components=5)
fit_PCA = pca.fit_transform(fit_SS)

result = model.predict(fit_PCA)
pd.DataFrame(result).to_csv("Results.csv", header=None, index=False)