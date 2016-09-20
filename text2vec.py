import pdb
import os

import json

import nltk
from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy
import scipy.io

import matplotlib.pyplot as plt

import preprocessing_config

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

'''Constructs word frequency matrix and returns it. The object type returned is
scipy sparse matrix. Further, the matrix and the features are stored in
word_freq_matrix.mtx and labels.dat respectively'''
def construct_word_freq_matrix(parsed_documents):
    
    '''Take the list of dictionaries as input and separate out document body,
    title, topics and places. The first two are tokeinzed and converted to
    features whereas the last two are converted to labels'''

    documents_as_string = []
    topics_labels = []
    places_labels = []

    for data_element in parsed_documents:
        try:
            document_string = ""

            if 'body' in data_element.keys():
                document_string += data_element['body'] 
            if 'title' in data_element.keys():
                document_string += " "+data_element['title']

            if 'topics' in data_element.keys():
                topics_labels.append(",".join(data_element['topics']))
            else:
                topics_labels.append("")

            if 'places' in data_element.keys():
                places_labels.append(",".join(data_element['places']))
            else:
                places_labels.append("")

            documents_as_string.append(document_string)

        except Exception as e:
            print "failed for document number %d" %count


    '''Initialize an object of CountVectorizer. We are using
    RegexTokenizerAndStemmer as tokenizer (defined above). Stop words as defined
    in the nltk are used directly.
    '''
    count_vectorizer = CountVectorizer(
                           tokenizer=RegexTokenizerAndStemmer(),
                           lowercase=True,
                           stop_words=nltk.corpus.stopwords.words('english'),
                           strip_accents='ascii',
                      )
    bigram_vectorizer = CountVectorizer(
                           tokenizer=RegexTokenizerAndStemmer(),
                           lowercase=True,
                           stop_words=nltk.corpus.stopwords.words('english'),
                           strip_accents='ascii',
                           ngram_range=(2,2)
                      )
    X_freq = count_vectorizer.fit_transform(documents_as_string)
    X_bigram_freq = bigram_vectorizer.fit_transform(documents_as_string)

    write_sparse_data_matrix_to_file(X_freq, "word_freq_matrix")
    write_sparse_data_matrix_to_file(X_bigram_freq, "bigram_freq_matrix")

    print "successfully saved word frequency matrix"

    save_features_to_file(count_vectorizer, "unigram")
    save_features_to_file(bigram_vectorizer, "bigram")

    save_labels_to_file(topics_labels, 'topics_labels.dat')
    save_labels_to_file(places_labels, 'places_labels.dat')

    return X_freq

def save_labels_to_file(labels, filename):
    f = open(os.path.join(preprocessing_config.output_data_dir, filename), 'w')
    f.write("\n".join(labels))
    f.close()

'''Take word frequency matrix as input and return tf-idf matrix'''
def construct_tf_idf_matrix(X_freq):
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_freq)

    output_dir = preprocessing_config.output_data_dir
    write_sparse_data_matrix_to_file(X_tfidf, "tf_idf_matrix")
    print "successfully saved tf-idf matrix"
    return X_tfidf

def save_features_to_file(vectorizer, prefix):
    features = vectorizer.get_feature_names()
    full_filename = os.path.join(preprocessing_config.output_data_dir,
                                 prefix + '_features.dat')
    f = open(full_filename, 'w')
    f.write("\n".join(features))
    f.close()


def write_sparse_data_matrix_to_file(X, filename):
    full_filename = os.path.join(preprocessing_config.output_data_dir, filename)
    scipy.io.mmwrite(full_filename, X)

def plot_freq_ranking_graph(X):
    total_word_freq = sorted(numpy.array(X.sum(0))[0], reverse=True)
    plt.xlabel("Log Ranking of Words")
    plt.ylabel("Log Words Frequency")
    plt.yscale('log', nonposy='clip')
    plt.xscale('log', nonposy='clip')
    plt.plot(range(1, len(total_word_freq)+1), total_word_freq)
    plt.savefig('plots/word_freq_graph.png', bbox_inches='tight')
    print "Plot indicating Zipf's law saved to plots/word_freq_graph.png"
    plt.close()
