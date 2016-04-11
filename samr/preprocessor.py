from nltk import pos_tag, PorterStemmer
from nltk.corpus import stopwords, wordnet
from samr import corpus
from nltk.stem.wordnet import WordNetLemmatizer

__author__ = 'espen1'
REMOVE_STOPWORDS = False
LEMMATIZE = False
STEMMING = False
BREAK_INTO_WORDS = True
def buildVocabulary(sentences):
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
            if REMOVE_STOPWORDS and word in stopWords:
                continue
            if(LEMMATIZE):
                if(morphy_tag.get(tags[i][1][:2]) != None):#Means that we can do some lemmatization
                    filteredWords.append(lmtzr.lemmatize(word, morphy_tag.get(tags[i][1][:2])))
                    continue
            if(STEMMING):
                filteredWords.append(stemmer.stem(word))
                continue
            filteredWords.append(word)
        if not BREAK_INTO_WORDS:
            corpus.append(" ".join(filteredWords))
        else:
            corpus.append(filteredWords)
    return corpus

def getPhrases(data):
    phrases = []
    for datapoint in data:
        phrases.append(datapoint.phrase)
    return phrases
def getLabels(data):
    labels = []
    for datapoint in data:
        labels.append(datapoint.sentiment)
    return labels

def getTrainingAndTestSplitOnSize(size):
    train, test = corpus.make_train_test_split("mySeed")
    trainingDataWithLabels, trainingDataShortWithLabels = getWordListsGreaterThan(size, train)
    testDataWithLabels, testDataShortWithLabels = getWordListsGreaterThan(size, test)
    return trainingDataWithLabels, trainingDataShortWithLabels, testDataWithLabels, testDataShortWithLabels
def getDataSplitOnSize(data, size):
    return getWordListsGreaterThan(size, data)

def getVocabularyOfSizeGreaterThan(size):
    train, test = corpus.make_train_test_split("mySeed")
    trainingDataWithLabels, dummy = getWordListsGreaterThan(size, train)
    testDataWithLabels, dummy = getWordListsGreaterThan(size, test)
    return trainingDataWithLabels, testDataWithLabels

def splitDataIntoShortAndLong(splitSize, data):
    shortSentences = []
    longSentences = []
    shortLabels = []
    longLabels = []
    longIndexes = []
    for i,(sentiment, datapoint) in enumerate(zip(getLabels(data), data)):
        if datapoint.phrase.count(" ") > splitSize:
            longSentences.append(datapoint)
            longLabels.append(sentiment)
            longIndexes.append(i)
        else:
            shortSentences.append(datapoint)
            shortLabels.append(sentiment)
    return shortSentences, shortLabels, longSentences, longLabels, longIndexes

def getWordListsGreaterThan(size, datapoints = None, data = None):
    if data == None: trainingWordsList = buildVocabulary(getPhrases(datapoints))
    else:
        sentences = getPhrases(data)

    labels = getLabels(datapoints)
    filteredList = []
    filteredShorterList = []
    for i, wordList in enumerate(trainingWordsList):
        if len(wordList) > size:
            filteredList.append((wordList, labels[i]))
        else:
            filteredShorterList.append((wordList, labels[i]))
    return filteredList, filteredShorterList
