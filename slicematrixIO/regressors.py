from core import BasePipeline
from utils import rando_name, r_squared
from uuid import uuid4
import pandas as pd
import numpy as np

# Random Forest Regressor pipeline #############################################################################################################################################3
class RFRegressorPipeline(BasePipeline):
    def __init__(self, name, n_trees = 8, client = None):
        params = {"n_trees": n_trees,}
        BasePipeline.__init__(self, name, "raw_rfr", client, params)

    def run(self, X, Y, model):
        return BasePipeline.run(self, X = X, Y = Y, model = model)

# Random Forest Regressor model; if pipeline == None then create a pipeline for use with this model
class RFRegressor():
    def __init__(self, X = None, Y = None, name = None, pipeline = None, n_trees = 8, client = None):
        self.client  = client
        self.type     = "raw_rfr"
        if X is not None and Y is not None:
            self.__full_init__(X, Y, name, pipeline, n_trees, client)
        else:
            self.__lazy_init__(name)

    def __full_init__(self, X, Y, name = None, pipeline = None, n_trees = 8, client = None):
        self.n_trees = n_trees 
        if name == None:
            name = rando_name()
        # else:
        #    todo: add feature to instantiate RFRegressor just from name
        #    i.e. an already created model
        self.name     = name
        self.X        = X
        self.Y        = Y
        self.pipeline = pipeline
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = RFRegressorPipeline(pipeline_name, n_trees, client)
        self.response = self.pipeline.run(X = self.X, Y = self.Y, model = self.name)
        try:
            # model will be key if success
            model = self.response['model']
            self.name = model.split("/")[-1]
        except:
            # something went wrong creating the model
            raise StandardError(self.response)

    # lazy loading for already persisted models
    def __lazy_init__(self, model_name):
        self.name     = model_name
        
    def predict(self, point):
        # todo: predict class given new point
        extra_params = {"features": point.values.tolist()}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "predict",
                                          extra_params = extra_params)
        try:
           return pd.DataFrame(response['predict'])
        except:
           raise StandardError(response)

    def score(self):
        Y_hat = self.predict(self.X)
        return r_squared(Y_hat, self.Y)

# K Nearest Neighbors regressor pipeline
class KNNRegressorPipeline(BasePipeline):
    def __init__(self, name, K = 5, kernel = "euclidean", algo = "auto", weights = "uniform", kernel_params = {}, client = None):
        params = {"k": K,
                  "kernel": kernel,
                  "algo": algo,
                  "weights": weights,
                  "kernel_params": kernel_params}
        BasePipeline.__init__(self, name, "raw_knn_regressor", client, params)

    def run(self, X, Y, model):
        return BasePipeline.run(self, X = X, Y = Y, model = model)

class KNNRegressor():
    def __init__(self, X = None, Y = None, name = None, pipeline = None, K = 5, kernel = "euclidean", algo = "auto", weights = "uniform", kernel_params = {}, client = None):
        self.client  = client
        self.type     = "raw_knn_regressor"
        if X is not None and Y is not None:
            self.__full_init__(X, Y, name, pipeline, K, kernel, algo, weights, kernel_params, client)
        else:
            self.__lazy_init__(name)

    def __full_init__(self, X, Y, name = None, pipeline = None, K = 5, kernel = "euclidean", algo = "auto", weights = "uniform", kernel_params = {}, client = None):
        self.X       = X
        self.Y       = Y
        if name == None:
            name = rando_name()
        self.name     = name
        self.pipeline = pipeline
        self.K        = K
        self.kernel   = kernel
        self.kernel_params = kernel_params
        self.algo     = algo
        self.weights  = weights
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = KNNRegressorPipeline(pipeline_name, K, kernel, algo, weights, kernel_params, client)
        self.response = self.pipeline.run(X = self.X, Y = self.Y, model = self.name)
        try:
            # model will be key if success
            model = self.response['model']
            self.name = model.split("/")[-1]
        except:
            # something went wrong creating the model
            raise StandardError(self.response)

    # lazy loading for already persisted models
    def __lazy_init__(self, model_name):
        self.name     = model_name

    def predict(self, point):
        extra_params = {"features": point.values.tolist()}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "predict",
                                          extra_params = extra_params)
        try:
           return pd.DataFrame(response['predict'])
        except:
           raise StandardError(response)

    def score(self):
        Y_hat = self.predict(self.X)
        return r_squared(Y_hat, self.Y)


#################################################################################################################################################################################
class KernelRidgeRegressorPipeline(BasePipeline):
    def __init__(self, name, kernel = "linear", alpha = 1.0, kernel_params = {}, client = None):
        params = {"kernel": kernel,
                  "alpha": alpha,
                  "kernel_params": kernel_params}
        BasePipeline.__init__(self, name, "raw_krr", client, params)

    def run(self, X, Y, model):
        return BasePipeline.run(self, X = X, Y = Y, model = model)

class KernelRidgeRegressor():
    def __init__(self, X = None, Y = None, name = None, pipeline = None, kernel = "linear", alpha = 1.0, kernel_params = {}, client = None):
        self.client  = client
        self.type     = "raw_krr"
        if X is not None and Y is not None:
            self.__full_init__(X, Y, name, pipeline, kernel, alpha, kernel_params, client)
        else:
            self.__lazy_init__(name)        

    def __full_init__(self, X, Y, name = None, pipeline = None, kernel = "linear", alpha = 1.0, kernel_params = {}, client = None):
        self.X       = X
        self.Y       = Y
        if name == None:
            name = rando_name()
        self.name     = name
        self.pipeline = pipeline
        self.kernel   = kernel
        self.alpha    = alpha        
        self.kernel_params = kernel_params
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = KernelRidgeRegressorPipeline(pipeline_name, kernel, alpha, kernel_params, client)
        self.response = self.pipeline.run(X = self.X, Y = self.Y, model = self.name)
        try:
            # model will be key if success
            model = self.response['model']
            self.name = model.split("/")[-1]
        except:
            # something went wrong creating the model
            raise StandardError(self.response)

    # lazy loading for already persisted models
    def __lazy_init__(self, model_name):
        self.name     = model_name

    def predict(self, point):
        extra_params = {"features": point.values.tolist()}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "predict",
                                          extra_params = extra_params)
        try:
           return pd.DataFrame(response['predict'])
        except:
           raise StandardError(response)

    def score(self):
        Y_hat = self.predict(self.X)
        return r_squared(Y_hat, self.Y)

