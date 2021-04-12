from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from .DefaultModels import GradientBoosting
import sklearn
import pandas as pd
import numpy as np

from sklearn.base import RegressorMixin, BaseEstimator


class SSModel(BaseEstimator, RegressorMixin):  # Scaleout-Speedup model

    def __init__(self, instance_count_index=0, regressor=None):
        self.regressor = regressor
        self.scales = []
        self.instance_count_index = instance_count_index  # index within features

    def preprocess(self, X, y):
        # Find biggest group of same features besides instance_count to learn from
        Xy = np.concatenate((X, y.reshape(-1,1)), axis=1)
        features = pd.DataFrame(Xy)
        indices = list(range(len(X[0])))
        indices.remove(self.instance_count_index)
        groups = features.groupby(by=indices)
        max_group = sorted(groups, key=lambda x:len(x[1]))[-1][1]
        X = max_group.iloc[:, 0].to_numpy().reshape((-1,1))
        y = max_group.iloc[:, -1]
        return X, y

    def fit(self, X, y):
        if X.shape[1] > 1:
            X, y = self.preprocess(X, y)

        self.min, self.max = X.min(), X.max()
        self.regressor.fit(X,y)

    def predict(self, X):
        rt_for_min_scaleout = self.regressor.predict(np.array([[self.min]]))
        # Make it a 2-dim array, as it is usually supposed to be
        rt = self.regressor.predict(X)[:, np.newaxis]
        # Replace scale-outs of more than self.max with pred for self.max
        # (poly3 curve does not continue as desired)
        rt[X.flatten() > self.max] = self.regressor.predict(np.array([[self.max]]))
        return (rt/rt_for_min_scaleout).flatten()


class OptimisticModel(BaseEstimator, RegressorMixin):

    def __init__(self, ibm, ssm):
        self.ssm= SSModel(regressor=ssm)
        self.ibm= ibm

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.instance_count_index = 0
        # Train scale-out speed-up model
        self.ssm.fit(X, y)
        scales = self.ssm.predict(X[:,[self.instance_count_index]])
        #print('scales', scales.shape)
        # Project all runtimes to expected runtimes at scaleout = min_scaleout
        y_projection = y/scales
        #print('yproj', y_projection.shape)
        # Train the inputs-behavior model on all inputs (not the instance_count)
        inputs = [i for i in range(X.shape[1]) if i != self.instance_count_index] or [0]
        self.ibm.fit(X[:,inputs], y_projection)

    def predict(self, X):
        X = np.array(X)
        instance_count = X[:, [self.instance_count_index]]
        inputs = list([i for i in range(X.shape[1]) if i != self.instance_count_index])
        m1 = self.ssm.predict(instance_count).flatten()
        m2 = self.ibm.predict(X[:,inputs]).flatten()
        y_pred = m1 * m2
        return y_pred


class BasicOptimisticModel(BaseEstimator, RegressorMixin):

    def __init__(self):
        polyreg3 = make_pipeline(PolynomialFeatures(3), LinearRegression())
        self.estimator = OptimisticModel(ibm=LinearRegression(), ssm=polyreg3)

        self.fit = self.estimator.fit
        self.predict = self.estimator.predict


class OptimisticGradientBoosting(BaseEstimator, RegressorMixin):

    def __init__(self):
        ssm = GradientBoosting(learning_rate=0.5, n_estimators=50)
        ibm = GradientBoosting(learning_rate=0.05, n_estimators=300)
        self.estimator = OptimisticModel(ibm=ibm, ssm=ssm)

        self.fit = self.estimator.fit
        self.predict = self.estimator.predict
