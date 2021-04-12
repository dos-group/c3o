import sklearn
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct


class GradientBoosting(BaseEstimator, RegressorMixin):

    def __init__(self, learning_rate=0.1, n_estimators=1000):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        regressor = GradientBoostingRegressor(learning_rate=self.learning_rate,
                                              n_estimators=self.n_estimators)
        estimator = Pipeline(steps=[
                ('ss', StandardScaler()),
                ('gb', regressor) ])

        self.fit = estimator.fit
        self.predict = estimator.predict


class ErnestModel(BaseEstimator, RegressorMixin):

    def _fmap(self, x):
        x = np.array(x)
        scaleout, problem_size = x[:,0], x[:,1]
        return np.c_[np.ones_like(scaleout),
                     problem_size/scaleout,
                     np.log(scaleout),
                     scaleout]

    def fit(self, x, y):
        X = self._fmap(x)
        y = np.array(y).flatten()
        self.coeff, _ = sp.optimize.nnls(X, y)

    def predict(self, x):
        X = self._fmap(x)
        return np.dot(X, self.coeff)

