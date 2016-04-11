import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn import svm
from unittest import TestCase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from samr import corpus
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from samr.preprocessor import buildVocabulary, getPhrases, getLabels
from samr.transformations import ExtractText, ClassifierSVM, Word2VecFeatureGenerator
# from tests import test_svm
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords

#Averages each vector
def buildWordVector(model, text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


class TestsWord2vec(TestCase):
    def setUp(self):
        self.train, self.test = corpus.make_train_test_split("mySeed")
        self.realTest = corpus.iter_test_corpus()
        self.corpus = buildVocabulary(getPhrases(self.train + self.test + self.realTest))
        # self.training = buildVocabulary(getPhrases(self.train))
        # self.testing = buildVocabulary(getPhrases(self.test))
        self.size = 50
        # self.trainingLabels = getLabels(self.train)
        # self.testLabels = getLabels(self.test)
    def test_creating_model(self):
        model = gensim.models.Word2Vec(sentences=self.corpus, min_count=3,workers=4,size=self.size,window=10)
        print(model.most_similar(positive=["good"], topn=10))
        model.save('../data/word2vec/model'+str(self.size))
        print("Model created with success")
    def test_loading_model(self):
        model = gensim.models.Word2Vec.load("../data/word2vec/model50")
        print(model.most_similar(positive=["good"], topn=10))
    def test_classification_using_word2vec(self):
        # model = gensim.models.Word2Vec.load("../data/word2vec/model50")
        model = Word2Vec(size=self.size, min_count=3)
        model.build_vocab(self.training)
        model.train(self.training)
        train_vecs = np.concatenate([buildWordVector(model, z, self.size) for z in self.training])
        train_vecs = scale(train_vecs)
        test_vecs = np.concatenate([buildWordVector(model, z, self.size) for z in self.testing])
        test_vecs = scale(test_vecs)
        lr = SGDClassifier(loss='log', penalty='l1')
        lr.fit(train_vecs, self.trainingLabels)
        probabilities = lr._predict_proba(test_vecs)
        print("Accuracy: %.2f "%lr.score(test_vecs, self.testLabels))
    def test_feature_generation(self):
        pipeline = make_pipeline(ExtractText(lowercase=True), Word2VecFeatureGenerator())
        print(pipeline.transform(self.train[:100]))


