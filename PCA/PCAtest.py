import pandas as pd
import numpy as np
from sklearn import decomposition

feature_file = pd.read_csv('luca_main_discovery_expr.csv')
feature_array = np.array(feature_file)[0:100]
label_file = pd.read_csv('luca_sample_metadata.csv')
label_array = np.array(label_file)[110:210]

feature_array = np.delete(feature_array, 0 ,axis = 1) 

for j in range(100):
    for i in range(0,7200):
        feature_array[j][i] = float(feature_array[j][i])

pca = decomposition.PCA(n_components=50)
new_array = pca.fit_transform(feature_array)
print(new_array.shape)

label_encoded_array = []
for i in label_array:
    title = i[6]
    if title.startswith("AD"):
        label_encoded_array.append(0)
    elif title.startswith("COID"):
        label_encoded_array.append(1)
    else:
        label_encoded_array.append(None)
        print("undefined label")

print(label_encoded_array)

