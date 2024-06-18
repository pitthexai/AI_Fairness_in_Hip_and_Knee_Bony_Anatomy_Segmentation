import numpy as np

class SkewedErrorRatio:
    def compute(self, scores):
        scores = 1 - scores

        max_score = np.max(scores)
        min_score = np.min(scores)

        return  max_score/min_score

class StandardDeviation:
    def compute(self, scores):
        return np.std(scores)