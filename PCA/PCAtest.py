import pandas as pd
import numpy as np
from sklearn import decomposition

csv_file = pd.read_csv('data_PCA.csv')
csv_file = np.array(csv_file)
test_array = csv_file[:,1:100]
test_array =test_array.T
for j in range(99):
    for i in range(7200):
        test_array[j][i] = float(test_array[j][i])

pca = decomposition.PCA(n_components=50)
new_array = pca.fit_transform(test_array)
print(new_array.shape)
print(pca.explained_variance_ratio_)

