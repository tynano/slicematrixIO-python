from connect import ConnectIO, Uploader

from bayesian_filters import KalmanOLS                                                     # lazy loading enabled
from distributions import KernelDensityEstimator, BasicA2D, IsolationForest                # lazy loading enabled                
from classifiers import KNNClassifier, PNNClassifier                                       # lazy loading enabled
from graphs import MinimumSpanningTree, CorrelationFilteredGraph, NeighborNetworkGraph     # lazy loading enabled 
from matrices import DistanceMatrix                                                        # lazy loading enabled
from matrix_models import MatrixMinimumSpanningTree, MatrixKernelPCA, MatrixAgglomerator   # lazy loading enabled
from manifolds import KernelPCA, LocalLinearEmbedder, LaplacianEigenmapper, Isomap         # lazy loading enabled
from regressors import KNNRegressor, RFRegressor, KernelRidgeRegressor                     # lazy loading enabled

class SliceMatrix():
    def __init__(self, api_key, region = "us-east-1"):
        self.api_key = api_key
        self.client  = ConnectIO(self.api_key, region = region)

    # bayesian filters ##########################################################################################################################################################
    # need to test this
    def KalmanOLS(self, dataset = None, init_alpha = None, init_beta = None, trans_cov = None, obs_cov = None, init_cov = None, optimizations = [], name = None, pipeline = None):
        return KalmanOLS(dataset = dataset,
                         init_alpha = init_alpha,
                         init_beta  = init_beta,
                         trans_cov  = trans_cov,
                         obs_cov  = obs_cov,
                         init_cov = init_cov,
                         optimizations = optimizations,
                         name     = name,
                         pipeline = pipeline,
                         client   = self.client)

    # classifiers ###############################################################################################################################################################
    def KNNClassifier(self, dataset = None, class_column = None, name = None, pipeline = None, K = 5, kernel = "euclidean", algo = "auto", weights = "uniform", kernel_params = {}):
        return KNNClassifier(dataset  = dataset, 
                             class_column = class_column,
                             name     = name, 
                             pipeline = pipeline, 
                             K        = K,
                             kernel   = kernel, 
                             algo     = algo, 
                             weights  = weights, 
                             kernel_params = kernel_params, 
                             client   = self.client)

    def PNNClassifier(self, dataset = None, class_column = None, name = None, pipeline = None, sigma = 0.1):
        return PNNClassifier(dataset  = dataset,
                             class_column = class_column,
                             name     = name,
                             pipeline = pipeline,
                             sigma    = sigma,
                             client   = self.client)

    
    # distributions #############################################################################################################################################################
    def KernelDensityEstimator(self, dataset = None, bandwith = "scott", kernel_params = {}, name = None, pipeline = None):
        return KernelDensityEstimator(dataset  = dataset,
                                      bandwidth = bandwith,
                                      name     = name,
                                      pipeline = pipeline,
                                      client   = self.client)

    def BasicA2D(self, dataset = None, retrain = True, name = None, pipeline = None):
        return BasicA2D(dataset  = dataset,
                        retrain  = retrain,
                        name     = name,
                        pipeline = pipeline,
                        client   = self.client)

    def IsolationForest(self, dataset = None, rate = 0.1, n_trees = 100, name = None, pipeline = None):
        return IsolationForest(dataset  = dataset,
                               rate  = rate,
                               n_trees = n_trees,
                               name     = name,
                               pipeline = pipeline,
                               client   = self.client)

    # regressors ################################################################################################################################################################
    def KNNRegressor(self, X = None, Y = None, K = 5, kernel = "euclidean", algo = "auto", weights = "uniform", kernel_params = {}, name = None, pipeline = None):
        return KNNRegressor(X = X,
                            Y = Y,
                            name     = name,
                            pipeline = pipeline,
                            K        = K,
                            kernel   = kernel,
                            algo     = algo,
                            weights  = weights,
                            kernel_params = kernel_params,
                            client   = self.client)


    def RFRegressor(self, X = None, Y = None, n_trees = 8, name = None, pipeline = None):
        return RFRegressor(X = X, 
                           Y = Y, 
                           n_trees = n_trees, 
                           name = name,
                           pipeline = pipeline,
                           client = self.client)

    def KernelRidgeRegressor(self, X = None, Y = None, kernel = "linear", alpha = 1.0, kernel_params = {}, name = None, pipeline = None):
        return KernelRidgeRegressor(X = X,
                                    Y = Y,
                                    kernel = kernel, 
                                    alpha  = alpha,
                                    kernel_params = kernel_params,
                                    name = name,
                                    pipeline = pipeline,
                                    client = self.client)
         
    # matrices #################################################################################################################################################################
    def DistanceMatrix(self, dataset = None, K = 5, kernel = "euclidean", kernel_params = {}, geodesic = False, name = None, pipeline = None):
        return DistanceMatrix(dataset = dataset,
                              K       = K,
                              kernel  = kernel,
                              kernel_params = kernel_params,
                              geodesic = geodesic,
                              name = name,
                              pipeline = pipeline,
                              client   = self.client)
    
    # matrix models ############################################################################################################################################################
    def MatrixMinimumSpanningTree(self, matrix = None, matrix_name = None, matrix_type = None, name = None, pipeline = None):
        return MatrixMinimumSpanningTree(matrix = matrix,
                                         matrix_name = matrix_name,
                                         matrix_type = matrix_type,
                                         name = name,
                                         pipeline = pipeline,
                                         client   = self.client)

    def MatrixKernelPCA(self, D = 2, matrix = None, matrix_name = None, matrix_type = None, name = None, pipeline = None):
        return MatrixKernelPCA(D = D, 
                               matrix = matrix,
                               matrix_name = matrix_name,
                               matrix_type = matrix_type,
                               name = name,
                               pipeline = pipeline,
                               client   = self.client)

    def MatrixAgglomerator(self, label_dataset = None, alpha = 0.1, matrix = None, matrix_name = None, matrix_type = None, name = None, pipeline = None):
        return MatrixAgglomerator(alpha = alpha, 
                                  label_dataset = label_dataset,
                                  matrix = matrix,
                                  matrix_name = matrix_name,
                                  matrix_type = matrix_type,
                                  name = name,
                                  pipeline = pipeline,
                                  client   = self.client)

    # manifolds ################################################################################################################################################################
    def KernelPCA(self, dataset = None, D = 2, kernel = "linear", alpha = 1.0, invert = False, kernel_params = {}, name = None, pipeline = None):
        return KernelPCA(dataset = dataset,
                         D = D,
                         kernel  = kernel,
                         kernel_params = kernel_params,
                         alpha = alpha,
                         invert = invert,
                         name = name,
                         pipeline = pipeline,
                         client   = self.client)

    def LocalLinearEmbedder(self, dataset = None, D = 2, K = 3, method = "standard", name = None, pipeline = None):                               
        return LocalLinearEmbedder(dataset = dataset,
                                   D = D,
                                   K = K,
                                   method = method, 
                                   name = name, 
                                   pipeline = pipeline,
                                   client   = self.client)

    def LaplacianEigenmapper(self, dataset = None, D = 2, affinity = "knn", K = 5, gamma = 1.0, name = None, pipeline = None):
        return LaplacianEigenmapper(dataset = dataset,
                                    D = D,
                                    K = K,
                                    gamma = gamma,
                                    name = name,
                                    pipeline = pipeline,
                                    client   = self.client)

    def Isomap(self, dataset = None, D = 2, K = 3, name = None, pipeline = None):
        return Isomap(dataset = dataset,
                      D = D,
                      K = K,
                      name = name,
                      pipeline = pipeline,
                      client   = self.client)

    # graphs ##################################################################################################################################################################
    def MinimumSpanningTree(self, dataset = None, corr_method = "pearson", name = None, pipeline = None):
        return MinimumSpanningTree(dataset = dataset,
                                   corr_method = corr_method, 
                                   name = name,
                                   pipeline = pipeline,
                                   client   = self.client)
    
    def CorrelationFilteredGraph(self, dataset = None, K = 3, name = None, pipeline = None):
        return CorrelationFilteredGraph(dataset = dataset,
                                        K = K,
                                        name = name,
                                        pipeline = pipeline,
                                        client   = self.client)

    def NeighborNetworkGraph(self, dataset = None, K = 3, kernel = "euclidean", name = None, pipeline = None):
        return NeighborNetworkGraph(dataset = dataset,
                                    K = K,
                                    kernel = kernel,
                                    name = name,
                                    pipeline = pipeline,
                                    client   = self.client)

