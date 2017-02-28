from core import BasePipeline
from utils import rando_name
from uuid import uuid4
import pandas as pd

#################################################################################################################################################################
class MinimumSpanningTreePipeline(BasePipeline):
    def __init__(self, name, corr_method = "pearson", client = None):
        params = {"corr_method": corr_method}
        BasePipeline.__init__(self, name, "raw_mst", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class MinimumSpanningTree():
    def __init__(self, dataset = None, name = None, pipeline = None, corr_method = "pearson", client = None):
        self.client  = client
        self.type     = "raw_mst"
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, corr_method, client)
        else:
            self.__lazy_init__(name)

    def __full_init__(self, dataset, name = None, pipeline = None, corr_method = "pearson", client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.corr_method = corr_method
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = MinimumSpanningTreePipeline(pipeline_name, corr_method, client)
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

    def nodes(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "nodes",
                                          extra_params = {})
        try:
           return response['nodes']
        except:
           raise StandardError(response)

    def rankLinks(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "rankLinks",
                                          extra_params = {})
        try:
           return response['rankLinks']
        except:
           raise StandardError(response)

    def edges(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "edges",
                                          extra_params = {})
        try:
           return response['edges']
        except:
           raise StandardError(response)

    def rankNodes(self, statistic = "closeness_centrality"):
        extra_params = {"statistic": statistic}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "rankNodes",
                                          extra_params = extra_params)
        try:
           return pd.DataFrame(response['rankNodes'], index = [statistic]).T.sort(columns = statistic)
        except:
           raise StandardError(response)

    def neighborhood(self, node):
        extra_params = {"node": node}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "neighborhood",
                                          extra_params = extra_params)
        try:
           return response['neighborhood']
        except:
           raise StandardError(response)


#################################################################################################################################################################
class CorrelationFilteredGraphPipeline(BasePipeline):
    def __init__(self, name, K = 3, client = None):
        params = {"K": K}
        BasePipeline.__init__(self, name, "raw_cfg", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class CorrelationFilteredGraph():
    def __init__(self, dataset = None, name = None, pipeline = None, K = 3, client = None):
        self.client  = client
        self.type     = "raw_cfg"
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, K, client)
        else:
            self.__lazy_init__(name)

    def __full_init__(self, dataset, name = None, pipeline = None, K = 3, client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.K        = K
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = CorrelationFilteredGraphPipeline(pipeline_name, K, client)
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

    def nodes(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "nodes",
                                          extra_params = {})
        try:
           return response['nodes']
        except:
           raise StandardError(response)

    def rankLinks(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "rankLinks",
                                          extra_params = {})
        try:
           return response['rankLinks']
        except:
           raise StandardError(response)

    def edges(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "edges",
                                          extra_params = {})
        try:
           return response['edges']
        except:
           raise StandardError(response)

    def rankNodes(self, statistic = "closeness_centrality"):
        extra_params = {"statistic": statistic}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "rankNodes",
                                          extra_params = extra_params)
        try:
           return pd.DataFrame(response['rankNodes'], index = [statistic]).T.sort(columns = statistic)
        except:
           raise StandardError(response)

    def neighborhood(self, node):
        extra_params = {"node": node}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "neighborhood",
                                          extra_params = extra_params)
        try:
           return response['neighborhood']
        except:
           raise StandardError(response)

#################################################################################################################################################################
class NeighborNetworkGraphPipeline(BasePipeline):
    def __init__(self, name, K = 3, kernel = "euclidean", client = None):
        params = {"K": K, 
                  "kernel": kernel}
        BasePipeline.__init__(self, name, "raw_knn_net", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class NeighborNetworkGraph():
    def __init__(self, dataset = None, name = None, pipeline = None, K = 3, kernel = "euclidean", client = None):
        self.client  = client
        self.type     = "raw_knn_net"
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, K, kernel, client)
        else:
            self.__lazy_init__(name)

    def __full_init__(self, dataset, name = None, pipeline = None, K = 3, kernel = "euclidean", client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.K        = K
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = NeighborNetworkGraphPipeline(pipeline_name, K, kernel, client)
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

    def nodes(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "nodes",
                                          extra_params = {})
        try:
           return response['nodes']
        except:
           raise StandardError(response)

    def rankLinks(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "rankLinks",
                                          extra_params = {})
        try:
           return response['rankLinks']
        except:
           raise StandardError(response)

    def edges(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "edges",
                                          extra_params = {})
        try:
           return response['edges']
        except:
           raise StandardError(response)

    def rankNodes(self, statistic = "closeness_centrality"):
        extra_params = {"statistic": statistic}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "rankNodes",
                                          extra_params = extra_params)
        try:
           return pd.DataFrame(response['rankNodes'], index = [statistic]).T.sort(columns = statistic)
        except:
           raise StandardError(response)

    def neighborhood(self, node):
        extra_params = {"node": node}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "neighborhood",
                                          extra_params = extra_params)
        try:
           return response['neighborhood']
        except:
           raise StandardError(response)


