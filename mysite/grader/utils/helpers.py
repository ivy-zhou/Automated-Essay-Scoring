import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
import math

import language_tool_python
from collections import Counter
import nltk
from nltk.corpus import stopwords

language_tool = language_tool_python.LanguageTool('en-US')


def accuracy(content):
    matches = language_tool.check(content)
    incorrect = len(matches)
    wc = word_count(content)
    return ((wc - incorrect) / wc)*100


def word_count(content):
    return len(nltk.word_tokenize(content))


def tense(content):
    tagged = nltk.pos_tag(content.split())
    counts = Counter(tag for word, tag in tagged)
    past = counts["VBD"] + counts["VBN"]
    present = counts["VB"] + counts["VBZ"] + counts["VBP"]
    future = counts["MD"]

    totverbs = past + present + future
    return (max(max(present, future), past) / totverbs) * 100


def semantic_score(content):
    scores = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences_essay = tokenizer.tokenize(content)
    mean = 0
    for i in range(len(sentences_essay)):
        for j in range(i+1, len(sentences_essay)):
            mean += semantic_similarity(sentences_essay[i], sentences_essay[j])

    return (mean/(len(sentences_essay)*(len(sentences_essay)-1)/2))*100


def semantic_similarity(sentence1, sentence2):
    # turn sentences into lists of words
    X_list = nltk.word_tokenize(sentence1)
    Y_list = nltk.word_tokenize(sentence2)

    sw = stopwords.words('english')  # create a list of non-keywords

    # turn lists of words into sets, and only include keywords
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}

    # form a set containing keywords of both strings
    l1 = []
    l2 = []
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0.0

    for i in range(len(rvector)):
        c += l1[i]*l2[i]
    geometric_mean = (sum(l1)*sum(l2))**0.5
    # return number of keywords that are shared between the two sentences divided by the geometric mean number of	  keywords in each sentence
    similarity = c / float(geometric_mean)
    return similarity


def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


def makeFeatureVec(words, model, num_features):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_features,), dtype="float32")
    num_words = 0.
    for word in words:
        if word in model:
            num_words += 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, num_words)
    return featureVec


def getAvgFeatureVecs(essays, model, num_features):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays), num_features), dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs
