from core import BasePipeline
from utils import rando_name
from uuid import uuid4
import pandas as pd

#################################################################################################################################################################
class KernelDensityEstimatorPipeline(BasePipeline):
    def __init__(self, name, bandwidth = "scott", client = None):
        params = {"bandwidth": bandwidth}
        BasePipeline.__init__(self, name, "raw_kde", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class KernelDensityEstimator():
    def __init__(self, dataset = None, name = None, pipeline = None, bandwidth = "scott", client = None):
        self.client  = client
        self.type     = "raw_kde"
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, bandwidth, client)
        else:
            self.__lazy_init__(name)

    def __full_init__(self, dataset, name = None, pipeline = None, bandwidth = "scott", client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.bandwidth = bandwidth
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = KernelDensityEstimatorPipeline(pipeline_name, bandwidth, client)
        self.response = self.pipeline.run(self.dataset, self.name)
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

    # update the model with new data and return updated state info
    def simulate(self, N = 1):
        extra_params = {"N": N}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "simulate",
                                          extra_params = extra_params)
        try:
           return response['simulate']
        except:
           raise StandardError(response)

    def hypercube(self, lower_bounds, upper_bounds):
        extra_params = {"lower_bounds": lower_bounds, 
                        "upper_bounds": upper_bounds}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "hypercube",
                                          extra_params = extra_params)
        try:
           return response['hypercube']
        except:
           raise StandardError(response)

#################################################################################################################################################################
class BasicA2DPipeline(BasePipeline):
    def __init__(self, name, retrain = True, client = None):
        params = {"retrain": retrain}
        BasePipeline.__init__(self, name, "basic_a2d", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class BasicA2D():
    def __init__(self, dataset = None, name = None, pipeline = None, retrain = True, client = None):
        self.client  = client
        self.type     = "basic_a2d"
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, retrain, client)
        else:
            self.__lazy_init__(name)
  
    def __full_init__(self, dataset, name = None, pipeline = None, retrain = True, client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.retrain  = retrain
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = BasicA2DPipeline(pipeline_name, retrain, client)
        self.response = self.pipeline.run(self.dataset, self.name)
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

    def getState(self):
        extra_params = {}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "getState",
                                          extra_params = extra_params)
        try:
           return response['getState']
        except:
           raise StandardError(response)

    def score(self, value):
        extra_params = {"value": value}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "score",
                                          extra_params = extra_params)
        try:
           return response['score']
        except:
           raise StandardError(response)

    def update(self, value):
        extra_params = {"value": value}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "update",
                                          extra_params = extra_params)
        try:
           return response['update']
        except:
           raise StandardError(response)

#################################################################################################################################################################
class IsolationForestPipeline(BasePipeline):
    def __init__(self, name, rate = 0.1, n_trees = 100, client = None):
        params = {"rate" : min(0.5, rate), 
                  "n_trees": n_trees}
        BasePipeline.__init__(self, name, "isolation_forest", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class IsolationForest():
    def __init__(self, dataset = None, name = None, pipeline = None, rate = 0.1, n_trees = 100, client = None):
        self.client  = client
        self.type     = "isolation_forest"
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, rate, n_trees, client)
        else:
            self.__lazy_init__(name)

    def __full_init__(self, dataset, name = None, pipeline = None, rate = 0.1, n_trees = 100, client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.rate  = rate
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = IsolationForestPipeline(pipeline_name, rate, n_trees, client)
        self.response = self.pipeline.run(self.dataset, self.name)
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

    def training_scores(self):
        extra_params = {}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "training_scores",
                                          extra_params = extra_params)
        try:
           return response['training_scores']
        except:
           raise StandardError(response)

    def score(self, points):
        extra_params = {"points": points}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "score",
                                          extra_params = extra_params)
        try:
           return response['score']
        except:
           raise StandardError(response)

