import nltk
from nltk import word_tokenize

'''Creating a class for regex tokenizing and stemming. This will be used by
   CountVectorizer for tokenizing. It uses small case alphabets for recognizing
   words. Punctuations are ignored. Words are stemmed using Porter Stemmer
   algorithm.
   '''
class RegexTokenizerAndStemmer(object):
    def __init__(self):
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.regex_tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-z]+')
    def __call__(self, doc_string):
        return [self.stemmer.stem(t) for t in self.regex_tokenizer.tokenize(
                                                doc_string.lower()
                                              )
               ]
