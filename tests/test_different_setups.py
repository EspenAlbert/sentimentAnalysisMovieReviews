import json
from unittest import TestCase

import gensim
import numpy as np
from sklearn.linear_model import SGDClassifier

from samr import corpus
from samr.predictor import PhraseSentimentPredictor



class TestSetups(TestCase):
    def setUp(self):
        self.train, self.test = corpus.make_train_test_split("mySeed")
        self.config = json.load(open("../data/model2.json"))
        # self.samples = len(self.train)
        self.samples = len(self.train)
    def runThroughSetup(self, **kwargs):
        predictor = PhraseSentimentPredictor(**kwargs)
        predictor.fit(self.train[:self.samples])
        return str(predictor.score(self.test))
    def test_with_preprocessing(self):
        preprocessingParams = [(False, False, False), (False, True, False), (True, False, False), (False, False, True), (False, True, True), (True, False, True)]
        labels = "Lemmatization  | Stemming | Stop words | Accuracy"
        print(labels)
        for params in preprocessingParams:
            print(params, self.runThroughSetup(preprocessor = True, useLemmatization=params[0], stemming=params[1], useStopWords=params[2], **self.config))
    def test_without_preprocessing(self):
        print("No preprocessing: ", self.runThroughSetup(preprocessor = False, **self.config))
    def test_with_word2vec(self):
        print("With word2vec: ", self.runThroughSetup(preprocessor = False, word2vecFeatures=True, **self.config))
    def test_all_classifiers(self):
        classifiers = ["sgd", "knn", "svc", "randomforest"]
        for classifier in classifiers:
            print(classifier, " score:", self.runThroughSetup(classifier=classifier))
    def test_with_multiple_svc(self):
        classifiers = ["sgd", "knn", "svc", "randomforest"]
        for classifier in classifiers:
            print(classifier, " score with svm features:", self.runThroughSetup(classifier=classifier, svm_features=True))
    def test_with_idf(self):
        print("With tfidf: ", self.runThroughSetup(useTfIdf=True, **self.config))
        print("Without tfidf: ", self.runThroughSetup(useTfIdf=False, **self.config))
    def test_with_splitModel(self):
        print("With split model: ", self.runThroughSetup(splitModel=True, **self.config))
        print("Without split model: ", self.runThroughSetup(splitModel=False, **self.config))
    def test_ensemble_classifier(self):
        print("With ensemble: ", self.runThroughSetup(classifier="ensemble"))
        print("Without ensemble: ", self.runThroughSetup(classifier="randomforest"))
