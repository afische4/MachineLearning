import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

filename = "p1dataset2021.txt"

df = pd.read_csv(filename, sep=" ", header=None)

arr = np.array(df)
#this is the array without the first 3 cols
arr = arr[:,3:]

# convert to a binary matrix

#get modes to compare
modes = np.array(df.mode())
modes = modes[0,3:]

bin_arr = np.zeros((len(arr), len(arr[0])))

for i in range(0, len(arr)):
  for j in range(0, len(arr[0])):
    if arr[i, j] == modes[j]:
      bin_arr[i,j] = 0
    else:
      bin_arr[i,j] = 1
#standardize the matrix
bin_arr = StandardScaler().fit_transform(bin_arr)
print(bin_arr)

#perform PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(bin_arr)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = principalDf

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

ax.scatter(finalDf.loc[:,'principal component 1'], finalDf.loc[:, 'principal component 2'])

ax.grid()
plt.show()
