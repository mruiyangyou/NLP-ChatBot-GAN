
import numpy as np
from nltk.stem.porter import PorterStemmer
import nltk
import ssl

'''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
'''

# tokenize
def tokenize(sentence):
    #return sentence.split(' ')
    return nltk.word_tokenize(sentence)


stemmer = PorterStemmer()
def stem(word):
    return stemmer.stem(word.lower())



def bag_of_words(sen, all_words):
    sentence_words = [stem(wrod) for wrod in sen]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, word in enumerate(all_words):
        if word in sentence_words:
            bag[i] += 1

    return bag



#sentence = "how are you"
#all_words = ['how','are','you','I','we']

#print(bag_of_words(sentence, all_words))

