from core import BasePipeline
from utils import rando_name
from uuid import uuid4
import pandas as pd

#################################################################################################################################################################
class KernelPCAPipeline(BasePipeline):
    def __init__(self, name, D = 2, kernel = "linear", alpha = 1.0, invert = False, kernel_params = {}, client = None):
        params = {"D": D,
                  "kernel": kernel,
                  "alpha": alpha,
                  "invert": invert,
                  "kernel_params": kernel_params}
        BasePipeline.__init__(self, name, "raw_kpca", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class KernelPCA():
    def __init__(self, dataset = None, name = None, pipeline = None, D = 2, kernel = "linear", alpha = 1.0, invert = False, kernel_params = {}, client = None):
        self.client  = client 
        self.type     = "raw_kpca"
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, D, kernel, alpha, invert, kernel_params, client)
        else:
            self.__lazy_init__(name)

    # full construction, i.e. start from zero and create it all...
    def __full_init__(self, dataset, name = None, pipeline = None, D = 2, kernel = "linear", alpha = 1.0, invert = False, kernel_params = {}, client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.D        = D
        self.kernel   = kernel
        self.alpha    = alpha
        self.invert   = invert
        self.kernel_params = kernel_params
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = KernelPCAPipeline(pipeline_name, D, kernel, alpha, invert, kernel_params, client)
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

    def inverse_embedding(self, nodes = True):
        nodes = self.nodes()
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "inverse_embedding",
                                          extra_params = {})
        try:
           return pd.DataFrame(response['inverse_embedding'], index = nodes)
        except:
           raise StandardError(response)

    def embedding(self, nodes = True):
        nodes = self.nodes()
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "embedding",
                                          extra_params = {})
        try:
           return pd.DataFrame(response['embedding'], index = nodes)
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
                                          method = "meta",
                                          extra_params = {})
        try:
           return response['meta']
        except:
           raise StandardError(response)

    def feature_names(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "feature_names",
                                          extra_params = {})
        try:
           return response['feature_names']
        except:
           raise StandardError(response)


#################################################################################################################################################################
class LocalLinearEmbedderPipeline(BasePipeline):
    def __init__(self, name, D = 2, K = 3, method = "standard", client = None):
        params = {"D": D,
                  "k": K,
                  "method": method}
        BasePipeline.__init__(self, name, "raw_lle", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class LocalLinearEmbedder():
    def __init__(self, dataset = None, name = None, pipeline = None, D = 2, K = 3, method = "standard", client = None):
        self.client  = client
        self.type     = "raw_lle"
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, D, K, method, client)
        else:
            self.__lazy_init__(name)        

    def __full_init__(self, dataset, name = None, pipeline = None, D = 2, k = 3, method = "standard", client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.D        = D
        self.k        = k
        self.method   = method
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = LocalLinearEmbedderPipeline(pipeline_name, D, k, method, client)
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

    def embedding(self, nodes = True):
        nodes = self.nodes()
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "embedding",
                                          extra_params = {})
        try:
           return pd.DataFrame(response['embedding'], index = nodes)
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

    def recon_error(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "recon_error",
                                          extra_params = {})
        try:
           return response['recon_err']
        except:
           raise StandardError(response)

    def meta(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "meta",
                                          extra_params = {})
        try:
           return response['meta']
        except:
           raise StandardError(response)

    def feature_names(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "feature_names",
                                          extra_params = {})
        try:
           return response['feature_names']
        except:
           raise StandardError(response)


#################################################################################################################################################################
class LaplacianEigenmapperPipeline(BasePipeline):
    def __init__(self, name, D = 2, affinity = "knn", K = 5, gamma = 1.0, client = None):
        params = {"D": D,
                  "K": K,
                  "affinity": affinity,
                  "gamma": gamma}
        BasePipeline.__init__(self, name, "raw_laplacian_eigenmap", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class LaplacianEigenmapper():
    def __init__(self, dataset = None, name = None, pipeline = None, D = 2, affinity = "knn", K = 5, gamma = 1.0, client = None):
        self.client  = client
        self.type     = "raw_laplacian_eigenmap"
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, D, affinity, K, gamma, client)
        else:
            self.__lazy_init__(name)        

    def __full_init__(self, dataset, name = None, pipeline = None, D = 2, affinity = "knn", K = 5, gamma = 1.0, client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.D        = D
        self.K        = K
        self.affinity = affinity
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = LaplacianEigenmapperPipeline(pipeline_name, D, affinity, K, gamma, client)
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

    def embedding(self, nodes = True):
        nodes = self.nodes()
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "embedding",
                                          extra_params = {})
        try:
           return pd.DataFrame(response['embedding'], index = nodes)
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
                                          method = "meta",
                                          extra_params = {})
        try:
           return response['meta']
        except:
           raise StandardError(response)

    def feature_names(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "feature_names",
                                          extra_params = {})
        try:
           return response['feature_names']
        except:
           raise StandardError(response)

    def affinity_matrix(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "affinity_matrix",
                                          extra_params = {})
        try:
           return response['affinity_matrix']
        except:
           raise StandardError(response)

#################################################################################################################################################################
class IsomapPipeline(BasePipeline):
    def __init__(self, name, D = 2, K = 3, client = None):
        params = {"D": D,
                  "K": K}
        BasePipeline.__init__(self, name, "raw_isomap", client, params)

    def run(self, dataset, model):
        return BasePipeline.run(self, dataset = dataset, model = model)

class Isomap():
    def __init__(self, dataset, name = None, pipeline = None, D = 2, K = 3, client = None):
        self.client  = client
        self.type     = "raw_isomap"
        if dataset is not None:
            self.__full_init__(dataset.T, name, pipeline, D, K, client)
        else:
            self.__lazy_init__(name)

    def __full_init__(self, dataset, name = None, pipeline = None, D = 2, K = 3, client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        self.D        = D
        self.K        = K
        self.type     = "raw_isomap"
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = IsomapPipeline(pipeline_name, D, K, client)
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

    def embedding(self, nodes = True):
        nodes = self.nodes()
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "embedding",
                                          extra_params = {})
        try:
           return pd.DataFrame(response['embedding'], index = nodes)
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

    def recon_error(self):
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "recon_error",
                                          extra_params = {})
        try:
           return response['recon_error']
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

    def search(self, point):
        extra_params = {"point": point}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "search",
                                          extra_params = extra_params)
        try:
           return response['search']
        except:
           raise StandardError(response)





