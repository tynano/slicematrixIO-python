from core import BasePipeline
from utils import rando_name
from uuid import uuid4
import pandas as pd

class DistanceMatrixPipeline(BasePipeline):
    def __init__(self, name, kernel = "euclidean", geodesic = False, K = 5, kernel_params = {}, client = None):
        params = {"k": K,
                  "kernel": kernel,
                  "kernel_params": kernel_params,
                  "geodesic": geodesic}
        BasePipeline.__init__(self, name, "dist_matrix", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class DistanceMatrix():
    def __init__(self, dataset = None, name = None, pipeline = None, K = 5, kernel = "euclidean", geodesic = False, kernel_params = {}, client = None):
        self.client  = client
        self.type     = "dist_matrix" 
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, K, kernel, geodesic, kernel_params, client)
        else:
            self.__lazy_init__(name)
    
    # full construction, i.e. start from zero and create it all...
    def __full_init__(self, dataset, name = None, pipeline = None, K = 5, kernel = "euclidean", geodesic = False, kernel_params = {}, client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.kernel   = kernel
        self.kernel_params = kernel_params
        self.K        = K
        self.geodesic = geodesic
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = DistanceMatrixPipeline(name = pipeline_name, 
                                                   K    = K, 
                                                   kernel = kernel,
                                                   geodesic = geodesic,
                                                   kernel_params = kernel_params, 
                                                   client = client)
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
        
    def rankDist(self, target, page = 0):
        # todo: predict class given new point
        extra_params = {"target": target,
                        "page": page}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "rankDist",
                                          extra_params = extra_params)
        try:
           return pd.DataFrame(response['rankDist'], index = ['distance']).T.sort(columns = "distance")
        except:
           raise StandardError(response)

    def getKeys(self):
        # todo: return r^2 for training model
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "getKeys",
                                          extra_params = {})
        try:
           return response['getKeys']
        except:
           raise StandardError(response)
