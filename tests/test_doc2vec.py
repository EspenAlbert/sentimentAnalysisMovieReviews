import random
import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
import numpy as np
from sklearn import svm
from unittest import TestCase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from samr import corpus
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from samr.preprocessor import buildVocabulary, getLabels, getPhrases
from samr.transformations import ExtractText, ClassifierSVM, Word2VecFeatureGenerator
# from tests import test_svm
from sklearn.linear_model import SGDClassifier
from tests import test_svm
from nltk.corpus import stopwords

def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(TaggedDocument(v, [label]))
    return labelized

def getVecs(model, taggedDocuments, size):
    vecs = [np.array(model.docvecs[z.tags]).reshape((1, size)) for z in taggedDocuments]
    return np.concatenate(vecs)

class TestDoc2vec(TestCase):
    def setUp(self):
        self.train, self.test = corpus.make_train_test_split("mySeed")
        self.samples = 50000
        self.xTrain = buildVocabulary(getPhrases(self.train[:self.samples]))
        self.xTest = buildVocabulary(getPhrases(self.test[:self.samples]))
        self.size = 150
        self.labelsTrain = getLabels(self.train)
        self.labelsTest = getLabels(self.test)
    def test_create_model(self):
        print("labeled sentence worked?")
        x_train = labelizeReviews(self.xTrain, 'TRAIN')
        x_test = labelizeReviews(self.xTest, 'TEST')
        model_dm = gensim.models.Doc2Vec(min_count=1, window=5, size=self.size, sample=1e-3, negative=5, workers=3)
        model_dbow = gensim.models.Doc2Vec(min_count=1, window=6, size=self.size, sample=1e-3, negative=5, dm=0, workers=3)
        sentences = x_train
        model_dm.build_vocab(sentences)
        model_dbow.build_vocab(sentences)
        # npArray = np.array(x_train)
        for epoch in range(10):
            print("Starting epoch:", str(epoch))
            # perm = np.random.permutation(npArray.shape[0])
            model_dm.train(random.sample(sentences, len(sentences)))
            model_dbow.train(random.sample(sentences, len(sentences)))
        # model_dm.train(x_train)
        train_vecs = getVecs(model_dm, x_train, self.size)
        train_vecs_dbow = getVecs(model_dbow, x_train, self.size)
        train_vecs_total = np.hstack((train_vecs, train_vecs_dbow))

        sentences = x_test
        for epoch in range(10):
            print("Starting epoch:", str(epoch))
            # perm = np.random.permutation(npArray.shape[0])
            model_dm.train(random.sample(sentences, len(sentences)))
            model_dbow.train(random.sample(sentences, len(sentences)))
        test_vecs = getVecs(model_dm, x_train, self.size)
        test_vecs_dbow = getVecs(model_dbow, x_train, self.size)
        test_vecs_total = np.hstack((test_vecs, test_vecs_dbow))
        lr = SGDClassifier(loss='log', penalty='l1')
        lr.fit(train_vecs_total, self.labelsTrain[:self.samples])

        print('Test Accuracy: %.2f'%lr.score(test_vecs_total, self.labelsTest[:self.samples]))
