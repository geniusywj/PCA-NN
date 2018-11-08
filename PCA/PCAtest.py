import pandas as pd
import numpy as np
from sklearn import decomposition

SAMPLE_SIZE = 600

feature_file = pd.read_csv('luca_main_discovery_expr.csv')
feature_array = np.array(feature_file)[0:SAMPLE_SIZE]
label_file = pd.read_csv('luca_sample_metadata.csv')
label_array = np.array(label_file)[0:SAMPLE_SIZE]

feature_array = np.delete(feature_array, 0 ,axis = 1) 

print(feature_array.shape)

for j in range(SAMPLE_SIZE):
    for i in range(0,7200):
        feature_array[j][i] = float(feature_array[j][i])

pca = decomposition.PCA(n_components=50)
new_array = pca.fit_transform(feature_array)
print(new_array.shape)

label_encoded_array = []
#TO DO: find correspond sample
for i in label_array:
    title = i[6]
    if title.startswith("AD"):
        label_encoded_array.append(0)
    elif title.startswith("COID"):
        label_encoded_array.append(1)
    elif title.startswith("NL"):
        label_encoded_array.append(2)
    elif title.startswith("SMCL"):
        label_encoded_array.append(3)
    elif title.startswith("SQ"):
        label_encoded_array.append(4)
    elif title.startswith("S0"):
        label_encoded_array.append(5)
    elif "adenocarcinoma" in title:
        label_encoded_array.append(6)
    elif "squamous" in title:
        label_encoded_array.append(7)
    elif "carcinoma" in title:
        label_encoded_array.append(8)
    elif title.startswith("LU"):
        label_encoded_array.append(9)    
    elif "small cell lung cancer" in title:
        label_encoded_array.append(10)
    elif "Mixture" in title:
        label_encoded_array.append(11)
    else:
        label_encoded_array.append(None)
        print("undefined label: ", title)

print(label_encoded_array)
