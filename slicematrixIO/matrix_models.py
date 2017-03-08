from core import BasePipeline
from utils import rando_name
from uuid import uuid4
import pandas as pd

###################################################################################################################################################################
class MatrixMinimumSpanningTreePipeline(BasePipeline):
    def __init__(self, name, client = None):
        params = {}
        BasePipeline.__init__(self, name, "matrix_mst", client, params)

    def run(self, model, matrix = None, matrix_name = None, matrix_type = None):
        if matrix is not None:
            return BasePipeline.run(self, matrix_name = matrix.name, matrix_type = matrix.type, model = model)
        else:
            return BasePipeline.run(self, matrix_name = matrix_name, matrix_type = matrix_type, model = model)

class MatrixMinimumSpanningTree():
    def __init__(self, matrix = None, matrix_name = None, matrix_type = None, name = None, pipeline = None, client = None):
        self.client  = client 
        self.type = "matrix_mst"
        if matrix is None and matrix_name is None and matrix_type is None:
            self.__lazy_init__(name)
        else:
            self.__full_init__(matrix, matrix_name, matrix_type, name, pipeline, client)
            
    def __full_init__(self, matrix, matrix_name = None, matrix_type = None, name = None, pipeline = None, client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.matrix   = matrix
        self.pipeline = pipeline
        self.matrix_name = matrix_name
        self.matrix_type = matrix_type
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = MatrixMinimumSpanningTreePipeline(name = pipeline_name, 
                                                              client = client)
        self.response = self.pipeline.run(model = self.name,
                                          matrix = self.matrix,
                                          matrix_name = self.matrix_name,
                                          matrix_type = self.matrix_type)
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

    def rankLinks(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "rankLinks",
                                          extra_params = {})
        try:
           return response['rankLinks']
        except:
           raise StandardError(response)

    def nodes(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "nodes",
                                          extra_params = {})
        try:
           return response['nodes']
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

    def neighborhood(self, node):
        extra_params = {"node": node}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "neighborhood",
                                          extra_params = extra_params)
        try:
           return pd.DataFrame(response['neighborhood'], index = ['weight']).T.sort(columns = "weight")
        except:
           raise StandardError(response)


###################################################################################################################################################################
class MatrixKernelPCAPipeline(BasePipeline):
    def __init__(self, name, D = 2, client = None):
        params = {"D": D}
        BasePipeline.__init__(self, name, "matrix_kpca", client, params)

    def run(self, model, matrix = None, matrix_name = None, matrix_type = None):
        if matrix is not None:
            return BasePipeline.run(self, matrix_name = matrix.name, matrix_type = matrix.type, model = model)
        else:
            return BasePipeline.run(self, matrix_name = matrix_name, matrix_type = matrix_type, model = model)

class MatrixKernelPCA():
    def __init__(self, matrix = None, matrix_name = None, matrix_type = None, name = None, pipeline = None, D = 2, client = None):
        self.client  = client
        self.type = "matrix_kpca"
        if matrix is None and matrix_name is None and matrix_type is None:
            self.__lazy_init__(name)
        else:
            self.__full_init__(matrix, matrix_name, matrix_type, name, pipeline, D, client)

    def __full_init__(self, matrix = None, matrix_name = None, matrix_type = None, name = None, pipeline = None, D = 2, client = None):
        self.D = D
        if name == None:
            name = rando_name()
        self.name     = name
        self.matrix   = matrix
        self.pipeline = pipeline
        self.matrix_name = matrix_name
        self.matrix_type = matrix_type
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = MatrixKernelPCAPipeline(D = D,
                                                    name = pipeline_name,
                                                    client = client)
        self.response = self.pipeline.run(model = self.name,
                                          matrix = self.matrix,
                                          matrix_name = self.matrix_name,
                                          matrix_type = self.matrix_type)
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

    def embedding(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "embedding",
                                          extra_params = {})
        try:
           return pd.DataFrame(response['embedding'], index = self.nodes())
        except:
           raise StandardError(response)

    def nodes(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "nodes",
                                          extra_params = {})
        try:
           return response['nodes']
        except:
           raise StandardError(response)

    def meta(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "get_meta",
                                          extra_params = {})
        try:
           return response['get_meta']
        except:
           raise StandardError(response)


###################################################################################################################################################################
class MatrixAgglomeratorPipeline(BasePipeline):
    def __init__(self, name, alpha = 0.1, client = None):
        params = {"alpha":alpha}
        BasePipeline.__init__(self, name, "matrix_agglomeration", client, params)

    def run(self, label_dataset, model, matrix = None, matrix_name = None, matrix_type = None):
        if matrix is not None:
            return BasePipeline.run(self, dataset = label_dataset, matrix_name = matrix.name, matrix_type = matrix.type, model = model)
        else:
            return BasePipeline.run(self, dataset = label_dataset, matrix_name = matrix_name, matrix_type = matrix_type, model = model)

class MatrixAgglomerator():
    def __init__(self, label_dataset = None, matrix = None, matrix_name = None, matrix_type = None, name = None, pipeline = None, alpha = 0.1, client = None):
        self.client  = client
        self.type     = "matrix_agglomeration"
        if label_dataset is None and matrix is None and matrix_name is None and matrix_type is None:
            self.__lazy_init__(name)
        else:
            self.__full_init__(label_dataset, matrix, matrix_name, matrix_type, name, pipeline, alpha, client)

    def __full_init__(self, label_dataset, matrix = None, matrix_name = None, matrix_type = None, name = None, pipeline = None, alpha = 0.1, client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.label_dataset  = label_dataset
        self.pipeline = pipeline
        self.matrix   = matrix
        self.matrix_name = matrix_name
        self.matrix_type = matrix_type
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = MatrixAgglomeratorPipeline(name  = pipeline_name,
                                                       alpha = alpha,
                                                       client = client)
        self.response = self.pipeline.run(label_dataset = self.label_dataset, 
                                          model = self.name,
                                          matrix = matrix,
                                          matrix_name = matrix_name,
                                          matrix_type = matrix_type)
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



