from  regex_tokenizer_and_stemmer import RegexTokenizerAndStemmer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np
import os
import scipy.io
import json

import parser
import minhashing_config

'''This script has functions to create k-shingles, based on characters or
words. These functions will be useful for doing minhashing'''


'''Create word based shingles. Input is a list of documents, each document is
represented as string. There are two outputs - A dictionary mapping 
shingles to a unique numid, and documents, which are represented as a set of
ids.'''

def create_word_shingles(parsed_documents, k):

    documents_as_string = []
    for data_element in parsed_documents:
        try:
            if 'body' in data_element.keys() and data_element['body'] != '':
                documents_as_string.append(data_element['body'])

        except Exception as e:
            print "failed for document number %d" %count


    '''Initialize an object of CountVectorizer. We are using
    RegexTokenizerAndStemmer as tokenizer. Stop words as defined
    in the nltk are used directly.
    '''
    shingler = CountVectorizer(
                           tokenizer=RegexTokenizerAndStemmer(),
                           lowercase=True,
                           stop_words=nltk.corpus.stopwords.words('english'),
                           strip_accents='ascii',
                           ngram_range=(k,k),
                           binary=True
                )

    shingled_docs = shingler.fit_transform(documents_as_string)
    shingled_docs = shingled_docs[shingled_docs.getnnz(1)>0]
    pdb.set_trace()
    return shingled_docs, shingler.get_feature_names()

def write_sparse_data_matrix_to_file(X, filename):
    full_filename = os.path.join(minhashing_config.shingles_path, filename)
    scipy.io.mmwrite(full_filename, X)

if __name__ == '__main__':
    num_shingles = 1
    print "Parsing..."
    with open('data/output/parsed_documents.txt') as f:
        parsed_documents = json.load(f)
    print "Constructing %d shingles..." %num_shingles
    shingled_docs, feature_names = \
        create_word_shingles(parsed_documents, num_shingles)
    write_sparse_data_matrix_to_file(
        shingled_docs, str(num_shingles) + "_shingles")
