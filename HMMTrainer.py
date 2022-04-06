from hmmlearn import hmm
import numpy as np


class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []
        self.model = hmm.GaussianHMM(
                        n_components=self.n_components,
                        covariance_type=self.cov_type,
                        n_iter=self.n_iter
                    )

    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    def get_score(self, input_data):
        return self.model.score(input_data)
