from core import BasePipeline
from utils import rando_name
from uuid import uuid4
import pandas as pd

#################################################################################################################################################################
class KalmanOLSPipeline(BasePipeline):
    def __init__(self, name, init_alpha = None, init_beta = None, trans_cov = None, obs_cov = None, init_cov = None, optimizations = [], client = None):
        params = {"init_alpha": init_alpha,
                  "init_beta": init_beta,
                  "trans_cov": trans_cov,
                  "obs_cov": obs_cov,
                  "init_cov": init_cov,
                  "optimizations": optimizations}
        BasePipeline.__init__(self, name, "kalman_ols", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class KalmanOLS():
    def __init__(self, dataset = None, name = None, pipeline = None, init_alpha = None, init_beta = None, trans_cov = None, obs_cov = None, init_cov = None, optimizations = [], client = None):
        self.client  = client
        self.type     = "kalman_ols"
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, init_alpha, init_beta, trans_cov, obs_cov, init_cov, optimizations, client)
        else:
            self.__lazy_init__(name)

    def __full_init__(self, dataset, name = None, pipeline = None, init_alpha = None, init_beta = None, trans_cov = None, obs_cov = None, init_cov = None, optimizations = [], client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.trans_cov = trans_cov
        self.obs_cov   = obs_cov
        self.init_cov  = init_cov
        self.optimizations = optimizations
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = KalmanOLSPipeline(pipeline_name, init_alpha, init_beta, trans_cov, obs_cov, init_cov, optimizations, client)
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
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "getState",
                                          extra_params = {})
        try:
           return response['getState']
        except:
           raise StandardError(response)

    def getTrainingData(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "getTrainingData",
                                          extra_params = {})
        try:
           return response['getTrainingData']
        except:
           raise StandardError(response)

    # update the model with new data and return updated state info
    def update(self, X, Y):
        extra_params = {"Y": Y, "X": X}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "update",
                                          extra_params = extra_params)
        try:
           return response['update']
        except:
           raise StandardError(response)
