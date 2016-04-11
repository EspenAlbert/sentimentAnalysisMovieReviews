import copy
from sklearn.metrics import accuracy_score
from samr import preprocessor


class MyEnsembler():
    def __init__(self, classifiers):
        self.classifiers = classifiers
    def fit(self, Z, Y):
        for classifier in self.classifiers:
            classifier.fit(Z, Y)
    def predict(self, Z):
        predictions = []
        for vector in Z:
            currentPrediction = []
            for classifier in self.classifiers:
                currentPrediction.append(classifier.predict(vector))
            predictions.append(max(currentPrediction))
        return predictions
    def score(self, phrases):
        """
        `phrases` should be a list of `Datapoint` instances.
        Return value is a `float` with the classification accuracy of the
        input.
        """
        pred = self.predict(phrases)
        return accuracy_score(preprocessor.getLabels(phrases), pred)