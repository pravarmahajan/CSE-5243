import numpy as np
import numpy.random as rnd
import sklearn.cross_validation as CV
import scipy.sparse

from sklearn import neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn.metrics
import sklearn.utils

from scipy.sparse import diags
from sklearn import tree
import os
import pdb
import preprocessing_config
import text2vec
import time


''' Filter out topics_labels and tf_idf_matrix points where topics_labels is not empty.'''
def filter_topics():
    my_tf_idf_matrix = text2vec.load_sparse_matrix_from_file("tf_idf_matrix")
    #my_tf_idf_matrix = text2vec.load_sparse_matrix_from_file("bigram_freq_matrix")
    #my_tf_idf_matrix = text2vec.load_sparse_matrix_from_file("word_freq_matrix")
    my_tf_idf_matrix = my_tf_idf_matrix.tocsr()
    with open(os.path.join(preprocessing_config.output_data_dir, "topics_labels.dat"), 'r') as f:
        my_topics_labels = f.readlines()

    a = []
    rows_to_keep = []
    idx = 0
    for label in my_topics_labels:
        label = label.strip()
        if label != "":
            a.append(label)
            rows_to_keep.append(idx)
        idx += 1

    X = my_tf_idf_matrix[rows_to_keep, :]
    Y = np.array([s.split(',') for s in a])
    return X, Y


'''Perform an 80-20 split on the filtered data. Output is a tuple in this
format: (X_train, X_test, Y_train, Y_test). The problem here is that the labels
have only one or two instances, because of which they appear only in the
training set or only in the test set. We will randomly split the data 80-20
then copy over points which exist only in test set and not in training set
and vice-versa. '''

def split_data_80_20(train_data, train_labels):
    X, Y = sklearn.utils.shuffle(train_data, train_labels, random_state =
                                    rnd.RandomState())

    n_rows_for_train = int(X.shape[0] * 0.8)
    X_train = X[:n_rows_for_train, :]
    X_test = X[n_rows_for_train:, :]
    Y_train = Y[:n_rows_for_train]
    Y_test = Y[n_rows_for_train: ]

    training_only_labels = set(item for sublist in Y_train for item in sublist)
    testing_only_labels = set(item for sublist in Y_test for item in sublist)

    rows_to_copy_over = []
    for row_num in range(Y_train.shape[0]):
        if not testing_only_labels >= set(Y_train[row_num]):
            rows_to_copy_over.append(row_num)
    
    X_test= scipy.sparse.vstack((X_test, X_train[rows_to_copy_over])).tocsr()
    Y_test = np.append(Y_test, Y_train[rows_to_copy_over])

    rows_to_copy_over = []
    for row_num in range(Y_test.shape[0]):
        if not training_only_labels >= set(Y_test[row_num]):
            rows_to_copy_over.append(row_num)

    X_train = scipy.sparse.vstack((X_train, X_test[rows_to_copy_over])).tocsr()
    Y_train = np.append(Y_train, Y_test[rows_to_copy_over])

    return X_train, Y_train, X_test, Y_test

''' MultiLabelBinarizer transform between iterable of iterables and a multilabel format. 
Although a list of sets or tuples is a very intuitive format for multilabel data, it is unwieldy to process.
This transformer converts between this intuitive format and the supported multilabel format: 
a (samples x classes) binary matrix indicating the presence of a class label.'''
def binarize_labels(labels, binarizer=None):
    if binarizer == None:
        binarizer = MultiLabelBinarizer()
        Y = binarizer.fit_transform(labels)
    else:
        Y = binarizer.transform(labels)

    return Y, binarizer

'''Perform KNN classification on the input dataset. Loops through different
values of k automatically and chooses the best value based on performance on the
input set. Prints the accuracy on the test set'''
def train_knn_classifier(train_data, train_labels):
    param_grid = [
        {'weights': ['uniform'], 'n_neighbors': np.logspace(0, 8, num=9, base=2)},
        {'weights': ['distance'], 'n_neighbors': np.logspace(0, 8, num=9, base=2)}
    ]   #tuning parameters, GridSearch will loop through these to get
        #the best set of params automatically

    time1 = time.time()
    clf = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid, cv=5,
                        scoring="f1_macro")
    clf.fit(train_data, train_labels)
    time2 = time.time()
    print "Time to train = %.2f secs" %(time2-time1)
    print "Best params obtained:"
    print clf.best_params_
    return clf

def test_knn_classifier(clf, test_data, test_labels):
    time1 = time.time()
    Y_pred = clf.predict(test_data)
    time2 = time.time()
    print "Accuracy score = %.2f" %sklearn.metrics.accuracy_score(test_labels, Y_pred)
    print "Precision score = %.2f" %sklearn.metrics.precision_score(test_labels,
                                    Y_pred, average = "macro")
    print "Recall score = %.2f" %sklearn.metrics.recall_score(test_labels,
                                    Y_pred, average = "macro")
    print "F1 score = %.2f" %sklearn.metrics.f1_score(test_labels,
                                    Y_pred, average = "macro")
    print "Time to test = %.2f secs" %(time2-time1)

    return Y_pred

'''Perform decision tree classification on the input dataset. Prints the time to train the data'''
def train_decision_tree_classifier(train_data, train_labels):
    time1 = time.time()
    clf  = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)

    clf.fit(train_data, train_labels)
    time2 = time.time()
    print "Time to train = %.2f secs" %(time2-time1)
    #print "Best params obtained:"
    #print clf.feature_importances_
    return clf

'''Tests the test data using training data. It compares current test_label with the observed labels. Accuracy, Precision, Recall and Time to test are also computed. '''
def test_decision_tree_classifier(clf, test_data, test_labels):
    time1 = time.time()
    Y_pred = clf.predict(test_data)
    time2 = time.time()
    print "Accuracy score = %.2f" %sklearn.metrics.accuracy_score(test_labels, Y_pred)
    print "Precision score = %.2f" %sklearn.metrics.precision_score(test_labels,
                                    Y_pred, average = "macro")
    print "Recall score = %.2f" %sklearn.metrics.recall_score(test_labels,
                                    Y_pred, average = "macro")
    print "F1 score = %.2f" %sklearn.metrics.f1_score(test_labels,
                                    Y_pred, average = "macro")
    print "Time to test = %.2f secs" %(time2-time1)

    return Y_pred

def main():
    X, Y = filter_topics()
    X_train, Y_train, X_test, Y_test = split_data_80_20(X, Y)
    Y_train_bin, binarizer = binarize_labels(Y_train)
    Y_test_bin, binarizer = binarize_labels(Y_test, binarizer)

    print "Runnning Decision Tree....."
    clf = train_decision_tree_classifier(X_train, Y_train_bin)

    Y_pred_bin = test_decision_tree_classifier(clf, X_test, Y_test_bin)
    Y_pred = binarizer.inverse_transform(Y_pred_bin)

    print "Running KNN...."
    clf = train_knn_classifier(X_train, Y_train_bin)

    Y_pred_bin = test_knn_classifier(clf, X_test, Y_test_bin)
    Y_pred = binarizer.inverse_transform(Y_pred_bin)

if __name__ == "__main__":
    main()
