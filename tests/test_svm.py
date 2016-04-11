from sklearn import svm
from unittest import TestCase
from sklearn.feature_extraction.text import TfidfVectorizer
from samr import corpus
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from samr.preprocessor import getLabels
from samr.transformations import ExtractText, ClassifierSVM


class TestSVM(TestCase):
    def setUp(self):
        self.train, self.test = corpus.make_train_test_split("mySeed")

    def test_prediction_of_svm(self):
        print(len(self.train))
        numberOfSamples = 10000
        pipeline = make_pipeline(ExtractText(True), TfidfVectorizer(min_df=5,
								 max_df = 0.8,
								 sublinear_tf=True,
								 use_idf=True,
                                     ngram_range=(1,1),
                                     max_features=1000))

        trainVectors = pipeline.fit_transform(self.train[:numberOfSamples])
        testVectors = pipeline.transform(self.test)
        self.assertEqual(trainVectors.shape[1], testVectors.shape[1])
        print(trainVectors.shape[1], "Feature count, starting learning...")

        classifier_linear = svm.SVC(kernel='linear')
        classifier_linear.fit(trainVectors, getLabels(self.train[:numberOfSamples]))
        prediction_linear = classifier_linear.predict(testVectors)
        print("Predictions: ", prediction_linear[:1000])
        print('1' and '0' and '3' and '4' in prediction_linear)
        print("Accuracy: ", accuracy_score(getLabels(self.test), prediction_linear))
        print("# of decision functions")
    def test_feature_generation(self):
        pipeline = make_pipeline(ExtractText(True), TfidfVectorizer(min_df=5,
								 max_df = 0.8,
								 sublinear_tf=True,
								 use_idf=True,
                                     ngram_range=(1,1),
                                     max_features=1000),
                                 ClassifierSVM()
                                 )
        vectors = pipeline.fit_transform(self.train, getLabels(self.train)).reshape(-1, 10)
        print(vectors[:10])
        self.assertEqual(vectors.shape[1], 10)
