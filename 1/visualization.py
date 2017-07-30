
# Data Preprocessing

# step1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
# step2: Importing the dataset
dataset = pd.read_csv('./DataSet/train.csv')
X_tr = dataset.iloc[:9500, 1:-1].values
Y_tr = dataset.iloc[:9500, -1].values

X_test =dataset.iloc[9501:,1:-1].values

# step3: Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #
imputer = imputer.fit(X_tr[:,:])
X_tr[:,:] = imputer.transform(X_tr[:,:])

# step4: Encoding categorical data
# 4.1 Encoding the Independent Variable
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = onehotencoder.fit_transform(X).toarray()
# # 4.2 Encoding the Dependent Variable
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
#
# # Step5: Splitting the dataset into the Training set and Test set

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

x_tr_set , x_test_set, y_train, y_test = train_test_split(X_tr, Y_tr, test_size = 0.2, random_state = 0)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(x_tr_set, y_train)
    print(clf.best_params_)
    print"--------------------------------------------------------"

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))



# clf = svm.SVC(C=0.0005, cache_size=200, coef0=0.1, degree=3, gamma= 'auto',
#     kernel='rbf', max_iter=-1, shrinking=True, tol=0.01, verbose=False)
# clf.fit(x_tr_set,y_train)
# y_hat = clf.predict(x_test_set)


# print str(accuracy_score(y_test,y_hat))
