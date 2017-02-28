# slicematrixIO-python
<h1>Python API for SliceMatrix-IO</h1>
<br>
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
