import os
from unittest import TestCase

from samr import preprocessor




class TestPreprocessor(TestCase):
    def setUp(self):
        preprocessor.REMOVE_STOPWORDS = False
    def test_make_longer_vocab(self):
        train, test = preprocessor.getVocabularyOfSizeGreaterThan(4)
        self.assertGreater(len(train[0][0]), 4)
        print(train[0])
        print(train[1])
        print(test[1])
        print(test[2])
    def test_lemmatization(self):
        preprocessor.LEMMATIZE = True
        wordsList = preprocessor.buildVocabulary(["I am demonstrating willpower"])
        self.assertTrue("demonstrating" not in wordsList[0])
        print(wordsList)
    def test_stemming(self):
        preprocessor.LEMMATIZE = False
        preprocessor.STEMMING = True
        wordsList = preprocessor.buildVocabulary(["I am demonstrating willpower"])
        self.assertTrue("demonstrating" not in wordsList[0])
        print(wordsList)
    def test_do_not_break_into_words(self):
        preprocessor.BREAK_INTO_WORDS = False
        sentences = preprocessor.buildVocabulary(["I am a great man", "I am demonstrating willpower", "This is fun"])
        self.assertTrue(isinstance(sentences[0], str))
        print(sentences)
    def test_split_vocab(self):
        train, trainShort, test, testShort = preprocessor.getTrainingAndTestSplitOnSize(4)
        self.assertGreater(len(train[0][0]), 4)
        self.assertLessEqual(len(trainShort[0][0]), 4)
        print(len(train), "# > |4|", len(trainShort), "# <=4")
        counts = self.getCounts(trainShort)
        print(counts)
        print(self.getCounts(train))
        print(train[0])
        print(train[1])
        print(test[1])
        print(test[2])
        print("Shorter:")
        print(trainShort[0])
        print(trainShort[1])
        print(testShort[1])
        print(testShort[2])

    def getCounts(self, trainShort):
        counts = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
        for instance in trainShort:
            counts[instance[1]] += 1
        return counts

