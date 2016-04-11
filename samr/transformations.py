"""
This module implements several scikit-learn compatible transformers, see
scikit-learn documentation for the convension fit/transform convensions.
"""
import os
import gensim
from nltk.corpus import stopwords, wordnet

import numpy
import re

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import fit_ovo, OneVsOneClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, PorterStemmer

import nltk


class StatelessTransform:
    """
    Base class for all transformations that do not depend on training (ie, are
    stateless).
    """
    def fit(self, X, y=None):
        return self
class Preprocessor(StatelessTransform):
    def __init__(self, removeStopWords = True, lemmatize = True, stem = False):
        self.stem = stem
        self.lemmatize = lemmatize
        self.removeStopWords = removeStopWords
    def getPhrases(self, data):
        phrases = []
        for datapoint in data:
            phrases.append(datapoint.phrase)
        return phrases
    def transform(self, X):
        """
        `X` is expected to be a list of `Datapoint` instances.
        Return value is a list of `str` instances in which words were tokenized
        and are separated by a single space " ". Optionally words are also
        lowercased depending on the argument given at __init__.
        """
        sentences = self.getPhrases(X)
        return self.buildVocabulary(sentences)
    def buildVocabulary(self, sentences):
        corpus = []
        stopWords = stopwords.words('english')
        lmtzr = WordNetLemmatizer()
        stemmer = PorterStemmer()
        morphy_tag = {'NN':wordnet.NOUN,'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV}
        for s in sentences:
            words = s.lower().split()
            tags = pos_tag(words)
            filteredWords = []
            for i, word in enumerate(words):
                if self.removeStopWords and word in stopWords:
                    continue
                if(self.lemmatize):
                    if(morphy_tag.get(tags[i][1][:2]) != None):#Means that we can do some lemmatization
                        filteredWords.append(lmtzr.lemmatize(word, morphy_tag.get(tags[i][1][:2])))
                        continue
                if(self.stem):
                    filteredWords.append(stemmer.stem(word))
                    continue
                filteredWords.append(word)
            corpus.append(" ".join(filteredWords))
        return corpus

class ExtractText(StatelessTransform):
    """
    This should be the first transformation on a samr pipeline, it extracts
    the phrase text from the richer `Datapoint` class.
    """
    def __init__(self, lowercase=False):
        self.lowercase = lowercase

    def transform(self, X):
        """
        `X` is expected to be a list of `Datapoint` instances.
        Return value is a list of `str` instances in which words were tokenized
        and are separated by a single space " ". Optionally words are also
        lowercased depending on the argument given at __init__.
        """
        it = (" ".join(nltk.word_tokenize(datapoint.phrase)) for datapoint in X)
        if self.lowercase:
            return [x.lower() for x in it]
        return list(it)


class ReplaceText(StatelessTransform):
    def __init__(self, replacements):
        """
        Replacements should be a list of `(from, to)` tuples of strings.
        """
        self.rdict = dict(replacements)
        self.pat = re.compile("|".join(re.escape(origin) for origin, _ in replacements))

    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        Return value is also a list of `str` instances with the replacements
        applied.
        """
        if not self.rdict:
            return X
        return [self.pat.sub(self._repl_fun, x) for x in X]

    def _repl_fun(self, match):
        return self.rdict[match.group()]


class MapToSynsets(StatelessTransform):
    """
    This transformation replaces words in the input with their Wordnet
    synsets[0].
    The intuition behind it is that phrases represented by synset vectors
    should be "closer" to one another (not suffer the curse of dimensionality)
    than the sparser (often poetical) words used for the reviews.

    [0] For example "bank": http://wordnetweb.princeton.edu/perl/webwn?s=bank
    """
    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        It returns a list of `str` instances such that the i-th element
        containins the names of the synsets of all the words in `X[i]`,
        excluding noun synsets.
        `X[i]` is internally tokenized using `str.split`, so it should be
        formatted accordingly.
        """
        return [self._text_to_synsets(x) for x in X]

    def _text_to_synsets(self, text):
        result = []
        for word in text.split():
            ss = nltk.wordnet.wordnet.synsets(word)
            result.extend(str(s) for s in ss if ".n." not in str(s))
        return " ".join(result)


class Densifier(StatelessTransform):
    """
    A transformation that densifies an scipy sparse matrix into a numpy ndarray
    """
    def transform(self, X, y=None):
        """
        `X` is expected to be a scipy sparse matrix.
        It returns `X` in a (dense) numpy ndarray.
        """
        return X.todense()


class ClassifierOvOAsFeatures:
    """
    A transformation that esentially implement a form of dimensionality
    reduction.
    This class uses a fast SGDClassifier configured like a linear SVM to produce
    a vector of decision functions separating target classes in a
    one-versus-rest fashion.
    It's useful to reduce the dimension bag-of-words feature-set into features
    that are richer in information.
    """
    def fit(self, X, y):
        """
        `X` is expected to be an array-like or a sparse matrix.
        `y` is expected to be an array-like containing the classes to learn.
        """
        self.classifier = OneVsOneClassifier(SGDClassifier(),n_jobs=-1).fit(X,numpy.array(y))
        return self

    def transform(self, X, y=None):
        """
        `X` is expected to be an array-like or a sparse matrix.
        It returns a dense matrix of shape (n_samples, m_features) where
            m_features = (n_classes * (n_classes - 1)) / 2
        """
        return self.classifier.decision_function(X)
class ClassifierSVM:
    def __init__(self, useOne = False, trainingSamples = 10000):
        self.useOne = useOne
        self.classifier = SVC(kernel='linear')
        self.trainingSamples = trainingSamples
    def fit(self, X, y):
        print(str(X.shape))
        self.classifier.fit(X[:self.trainingSamples], y[:self.trainingSamples])
        return self
    def transform(self, X, y=None):
        return self.classifier.decision_function(X)
class Word2VecFeatureGenerator(StatelessTransform):
    def __init__(self, fileLocation = os.path.join(os.path.dirname(__file__), "../data/word2vec")):
        print("File location: " + fileLocation)
        print("Loading Model..may take some time..please wait!")
        self.model = gensim.models.Word2Vec.load(fileLocation + "/model50")
        print("Loading model complete")
        print(self.model.most_similar(positive=["good"], topn=10))
        self.vocabulary = ["happy", "positive", "negative", "cool", "nice", "love", "hate", "not", "poor", "good", "ugly", "handsome", "fail", 'gay','suck', 'glad','bastard','fat', 'inspire', 'quality']
    # def transform(self, X, y=None):
    def transform(self, X, y=None):
        return [self.findSimilarity(sentence) for sentence in X]
    def findSimilarity(self, sentence):
        vocabularyScores = []
        for similarityWord in self.vocabulary:
            simScores = []
            for word in sentence.split():
                try:
                    if len(word) < 2: continue
                    simScores.append(self.model.similarity(similarityWord, word))
                except KeyError:
                    pass
            vocabularyScores.append(self.average(simScores))
        return vocabularyScores

    def average(self, listOfValues):
        if len(listOfValues) == 0: return 0
        return sum(listOfValues) / len(listOfValues)