import sklearn.cross_validation as CV

'''Perform an 80-20 split on the input data. Output is a tuple in this
format: (X_train, X_test, Y_train, Y_test)'''
def split_data_80_20(train_data, train_labels):
    return CV.train_test_split(train_data, train_labels, test_size=0.8)
