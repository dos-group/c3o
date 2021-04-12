from sklearn.model_selection import (LeaveOneOut, cross_val_score)
from sklearn.metrics import (make_scorer, mean_absolute_error as mae)

from .DefaultModels import (ErnestModel, GradientBoosting)
from .CustomModels import (BasicOptimisticModel, OptimisticGradientBoosting)


models = ((BasicOptimisticModel, {}),
          (OptimisticGradientBoosting, {}),
          (GradientBoosting, {}),
          (ErnestModel, {}))

scorer = make_scorer(mae)


class Predictor:

    def __init__(self):
        self.all_models = [model(**kwargs) for model, kwargs in models]
        self.chosen_model = None
        self.model_name = None
        self.predict = None

    def fit(self, X, y):
        """
        Choose and train the model with lowest expected error
        based on cross-validation
        """
        def average(iterable): return sum(iterable)/len(iterable)

        average_error_scores = []
        err_dict = {}
        for model in self.all_models:
            cv = LeaveOneOut().split(X)
            error_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
            err_dict[model] = error_scores
            average_error_scores.append((average(error_scores), model))

        self.chosen_model = min(average_error_scores, key=lambda x: x[0])[1]
        self.error_score = min(average_error_scores, key=lambda x:x[0])[0]
        self.training_errors = err_dict[self.chosen_model]

        self.chosen_model.fit(X, y)
        self.predict = self.chosen_model.predict