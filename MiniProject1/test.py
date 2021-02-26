from sklearn import datasets
import pandas as pd
 
 iris = datasets.load_iris()
 df = pd.DataFrame(iris.data, columns=iris.feature_names)
 df['Target'] = iris.get('target')
 df.head()
 # Create features and target datasets
 features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
 X = df[features].values
 y = df['Target'].values
 from sklearn.preprocessing import StandardScaler

 # Standardize the features
 X = StandardScaler().fit_transform(X)

 # Preview X
 pd.DataFrame(data=X, columns=features).head()
 # Import PCA from sklearn
 from sklearn.decomposition import PCA

 # Instantiate PCA
 pca = PCA(n_components=2)

 # Fit PCA to features
 principalComponents = pca.fit_transform(X)
 # Create a new dataset from principal components 
 df = pd.DataFrame(data = principalComponents, 
                   columns = ['PC1', 'PC2'])
target = pd.Series(iris['target'], name='target')
result_df = pd.concat([df, target], axis=1)
result_df.head(5)

# Visualize Principal Components with a scatter plot
fig = plt.figure(figsize = (12,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('First Principal Component ', fontsize = 15)
ax.set_ylabel('Second Principal Component ', fontsize = 15)
ax.set_title('Principal Component Analysis (2PCs) for Iris Dataset', fontsize = 20)

targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = iris['target'] == target
    ax.scatter(result_df.loc[indicesToKeep, 'PC1'], 
      result_df.loc[indicesToKeep, 'PC2'], 
      c = color, 
      s = 50)
ax.legend(targets)
ax.grid()
