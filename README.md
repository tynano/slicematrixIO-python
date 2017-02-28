# slicematrixIO-python
<h1>Python API for SliceMatrix-IO</h1>
<br>
<h3>Introduction</h3>
<p>SliceMatrix is the next generation of machine intelligence eco-system. SliceMatrix provides an end-to-end machine intelligence Platform as a Service (PaaS) so that users can seamlessly and easily develop machine intelligent applications and systems that are hosted directed on the cloud.</p>
<p>IO makes it trivially easy for any developer to access cutting edge machine learning algorithms. It is the first platform to commercialize manifold learning technology.</p>
<p>For example, the following code snipper fits a KernelPCA model to intraday price data for over 500 financial instruments. This is an example of manifold learning, a family of algorithms which discovers the underlying structure of high dimensional data. Manifold learning is designed to handle even highly nonlinear data structures such as financial, social, and medical datasets.</p>
```
from slicematrixIO import SliceMatrix

api_key = "<your api key>"
sm = SliceMatrix(api_key)

import pandas as pd
import numpy as np

df = pd.read_csv("intraday_prices.csv", index_col = 0)
df = np.log(df).diff().dropna()

raw_kpca = sm.KernelPCA(dataset = df.T, D = 3, invert = True)

embedding = raw_kpca.embedding()

print embedding.tail(20)
```
<p>The foundation of SliceMatrix is the graph model: this is a powerful mathematical abstraction which can represent any entity to entity relationship. For example, there is mounting evidence that there are systematic differences in returns between highly connected and outlier stock symbols within the S&P 500. One could easily rank stocks by their systematic importance and construct appropriate portfolios using one of SliceMatrix's graph algorithms, such as the Minimum Spanning Tree</p>
```
# calculate pairwise distances between all trading symbols
corr_mtx = sm.DistanceMatrix(dataset = df.T, geodesic = True, K = 3, kernel = "correlation")

corr_mst  = sm.MatrixMinimumSpanningTree(matrix = corr_mtx) # load directly from object

centrality = corr_mst.rankNodes(statistic = "closeness_centrality")

print centrality.head(20)

# output
"""
      closeness_centrality
BSX               0.032743
EFX               0.032743
BCR               0.033849
XRAY              0.034627
COW               0.034990
SYK               0.035025
SJM               0.035420
BMY               0.035866
IFF               0.036205
QCOM              0.036251
AVGO              0.036256
VAR               0.036279
QRVO              0.036279
NOC               0.036719
WLTW              0.037193
ATVI              0.037562
ADP               0.037613
CTAS              0.037623
APH               0.037623
PFE               0.037638

"""

```
<p>In addition, SliceMatrix provides foundational machine learning models such as classifiers and regressors. The next example fits a K-Nearest Neighbors Classifier to the statlog shuttle dataset:</p>
```
training_data = pd.read_csv("shuttle.trn",index_col = 0)
testing_data  = pd.read_csv("shuttle.tst",index_col = 0)

# train a K Nearest Neighbors Classifier where K = 10
knn = sm.KNNClassifier(K = 10, dataset = training_data, class_column = "class")

# get the % correct in the training set
print knn.score()

# now let's have fun...
# create 15,000 new out of sample predictions from the testing dataset (aka validation set)
testing_predictions = []
chunk = 1000
cindex = 0
while cindex < testing_data.shape[0]:
    eindex = cindex + chunk
    print cindex, eindex
    c_features = testing_data.drop("class", axis = 1).values[cindex:eindex].tolist()
    c_preds = knn.predict(c_features)
    testing_predictions.extend(c_preds)
    cindex += chunk

# compare predictions with ground truth
pct_correct = 1. * np.sum(np.equal(testing_predictions, testing_data['class'])) / len(testing_predictions)
print pct_correct #output => 0.99924137931
```
<h3>Installation</h3>
<p>The easiest way is to use pip:</p>
```
pip install slicematrixIO
````
