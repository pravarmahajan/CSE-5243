import sklearn.cross_validation as CV
import numpy.random as rnd

from sklearn import neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

'''Perform an 80-20 split on the input data. Output is a tuple in this
format: (X_train, X_test, Y_train, Y_test)'''
def split_data_80_20(train_data, train_labels):
    return CV.train_test_split(train_data, train_labels, test_size=0.8,
    random_state=rnd.RandomState())

'''Perform KNN classification on the input dataset. Loops through different
values of k automatically and chooses the best value based on performance on the
input set. Prints the accuracy on the test set'''
def classifiy_knn(train_data, train_labels, test_data, test_labels):
    param_grid = [
        {'weights': ['uniform', 'distance']},
        {'n_neighbors': np.logspace(0, 8, num=9, base=2)}
    ]   #tuning parameters, GridSearch will loop through these to get
        #the best set of params automatically

    mlb = MultiLabelBinarizer()
    X_train = train_data
    X_test = test_data
    Y_train = mlb.fit_trainsform(train_labels)
    Y_test = mlb.transform(test_labels)

    clf = GridSearchCV(neighbors.KNeighboursClassifier(), param_grid, cv=5, njobs=-1)
    clf.fit(X_train, Y_train)
    print clf.best_params_
    print clf.grid_scores_.mean_score

    Y_pred = clf.predict(X_test)
    print classification_report(Y_test, Y_pred, target_names=mlb.classes_)
