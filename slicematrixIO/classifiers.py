from core import BasePipeline
from utils import rando_name
from uuid import uuid4

# K Nearest Neighbors classifier pipeline
class KNNClassifierPipeline(BasePipeline):
    def __init__(self, name, K = 5, kernel = "euclidean", algo = "auto", weights = "uniform", kernel_params = {}, client = None):
        params = {"k": K,
                  "kernel": kernel,
                  "algo": algo,
                  "weights": weights,
                  "kernel_params": kernel_params}
        BasePipeline.__init__(self, name, "raw_knn_classifier", client, params)

    def run(self, dataset, model, class_column):
        return BasePipeline.run(self, dataset = dataset, model = model, extra_params = {"class_column": class_column})

# K Nearest Neighbors classifier model; if pipeline == None then create a pipeline for use with this model
class KNNClassifier():
    def __init__(self, dataset = None, class_column = None, name = None, pipeline = None, K = 5, kernel = "euclidean", algo = "auto", weights = "uniform", kernel_params = {}, client = None):
        self.client  = client 
        self.type     = "raw_knn_classifier"
        if dataset is not None:
            self.__full_init__(dataset, class_column, name, pipeline, K, kernel, algo, weights, kernel_params, client)
        else:
            self.__lazy_init__(name)     
   
    def __full_init__(self, dataset, class_column, name = None, pipeline = None, k = 5, kernel = "euclidean", algo = "auto", weights = "uniform", kernel_params = {}, client = None):
        self.class_column = class_column
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = KNNClassifierPipeline(pipeline_name, k, kernel, algo, weights, kernel_params, client)
        self.response = self.pipeline.run(self.dataset, self.name, self.class_column)
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
        extra_params = {"features": point}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "predict",
                                          extra_params = extra_params)
        try:
           return response['predict']
        except:
           raise StandardError(response)

    def score(self):
        # todo: return r^2 for training model
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "score",
                                          extra_params = {})
        try:
           return response['score']
        except:
           raise StandardError(response)

    def training_preds(self):
        # todo return predictions for training model
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "training_preds",
                                          extra_params = {})
        try:
           return response['training_preds']
        except:
           raise StandardError(response)

    def training_data(self):
        # todo return predictions for training model
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "training_data",
                                          extra_params = {})
        try:
           return response['training_data']
        except:
           raise StandardError(response)

# probalistic neural network classifier pipeline
class PNNClassifierPipeline(BasePipeline):
    def __init__(self, name, sigma = 0.1, client = None):
        params = {"sigma": sigma}
        BasePipeline.__init__(self, name, "raw_pnn", client, params)

    def run(self, dataset, model, class_column):
        return BasePipeline.run(self, dataset = dataset, model = model, extra_params = {"class_column": class_column})

# probalistic neural network classifier model; if pipeline == None then create a pipeline for use with this model
class PNNClassifier():
    def __init__(self, dataset, class_column, name = None, pipeline = None, sigma = 0.1, client = None):
        self.client  = client
        self.class_column = class_column
        self.type     = "raw_pnn"
        if dataset is not None:
            self.__full_init__(dataset, name, pipeline, sigma, client)
        else:
            self.__lazy_init__(name)

    def __full_init__(self, dataset, name = None, pipeline = None, sigma = 0.1, client = None):
        if name == None:
            name = rando_name()
        self.name     = name
        self.dataset  = dataset
        self.pipeline = pipeline
        if self.pipeline == None:
            pipeline_name = rando_name()
            self.pipeline = PNNClassifierPipeline(pipeline_name, sigma, client)
        self.response = self.pipeline.run(self.dataset, self.name, self.class_column)
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
        extra_params = {"features": point}
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "predict",
                                          extra_params = extra_params)
        try:
           return response['predict']
        except:
           raise StandardError(response)

    def score(self):
        # todo: return r^2 for training model
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "score",
                                          extra_params = {})
        try:
           return response['score']
        except:
           raise StandardError(response)

    def training_preds(self):
        # todo return predictions for training model
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "training_preds",
                                          extra_params = {})
        try:
           return response['training_preds']
        except:
           raise StandardError(response)

    def training_data(self):
        # todo return predictions for training model
        response = self.client.call_model(model  = self.name,
                                          type   = self.type,
                                          method = "training_data",
                                          extra_params = {})
        try:
           return response['training_data']
        except:
           raise StandardError(response)


