#Imported all the needed modules for the program
from tweepy import OAuthHandler
from tweepy import API
import pandas as pd

#Imported twitter credentials file
import credentials

# Importing preprocess from our module to pre-process the raw tweet text into useful information
import PreProcess
#created object of preprocess module
pp = PreProcess


#importing the needed modules to be used in machine learning from SkLearn(https://scikit-learn.org/)

# Importing Algorithms From Naive Bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

# Importing Algorithms From Linear Model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# Importing Algorithms Support Vector Machine
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# Importing Algorithm tree
from sklearn.tree import DecisionTreeClassifier

# Importing Algorithms From Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

# Importing Algorithms From Ensemble
from sklearn.ensemble import RandomForestClassifier


from sklearn.feature_extraction.text import CountVectorizer # This feaure Help in coverting Training Data to sparse matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


# Naive Bayes classification Algorithm

    #Multinomial Naive Bayes Algorithm

def MultinomialNBAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec):
    print("Multinomial Naive Bayes")
    mnb = MultinomialNB()
    #Fit function is used to fit estimator and Once the estimator is fitted, it can be used for predicting target values of new data
    mnb.fit(x_train_vec, y_train)
    # predict classes of the training data
    y_predict_class = mnb.predict(x_test_vec)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_predict_class))
    print('Accuracy Score :', accuracy_score(y_test, y_predict_class))
    print('ROC(Receiver Operating Characteristic) and AUC(Area Under Curve)', roc_auc_score(y_test, y_predict_class))
    print('Average Precision Score:', average_precision_score(y_test, y_predict_class))
    # predict classes of Fetched tweets data
    if mnb.predict(tweetvec) == [1]:
        return "Positive"
    else:
        return "Negative"


    #Bernoulli Naive Bayes Algorithm

def BernoulliNaiveBayesAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec):
    print("Bernoulli Naive Bayes")
    bnb = BernoulliNB()
    bnb.fit(x_train_vec, y_train)
    y_predict_class = bnb.predict(x_test_vec)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_predict_class))
    print('Accuracy Score :', accuracy_score(y_test, y_predict_class))
    print('ROC(Receiver Operating Characteristic) and AUC(Area Under Curve)', roc_auc_score(y_test, y_predict_class))
    print('Average Precision Score:', average_precision_score(y_test, y_predict_class))
    if bnb.predict(tweetvec) == [1]:
        return "Positive"
    else:
        return "Negative"


# Linear Model classification Algorithm

    #stochastic gradient descent Classifier Algorithm

def SGDClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec):
    print("stochastic gradient descent Classifier")
    sgd = SGDClassifier()
    sgd.fit(x_train_vec, y_train)
    y_predict_class = sgd.predict(x_test_vec)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_predict_class))
    print('Accuracy Score :', accuracy_score(y_test, y_predict_class))
    print('ROC(Receiver Operating Characteristic) and AUC(Area Under Curve)', roc_auc_score(y_test, y_predict_class))
    print('Average Precision Score:', average_precision_score(y_test, y_predict_class))
    if sgd.predict(tweetvec) == [1]:
        return "Positive"
    else:
        return "Negative"


    # Logistic Regression algotithm

def LogisticRegressionAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec):
    print("Logistic Regression")
    lr = LogisticRegression(multi_class='ovr', solver='liblinear')
    lr.fit(x_train_vec, y_train)
    y_predict_class = lr.predict(x_test_vec)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_predict_class))
    print('Accuracy Score :', accuracy_score(y_test, y_predict_class))
    print('ROC(Receiver Operating Characteristic) and AUC(Area Under Curve)', roc_auc_score(y_test, y_predict_class))
    print('Average Precision Score:', average_precision_score(y_test, y_predict_class))
    if lr.predict(tweetvec) == [1]:
        return "Positive"
    else:
        return "Negative"


# Support Vector Machine classification Algorithm

    #Support Vector Classifier Algorithm

def SupportVectorClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec):
    print("Support Vector Classifier")
    svc = SVC(kernel='linear')
    svc.fit(x_train_vec, y_train)
    y_predict_class = svc.predict(x_test_vec)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_predict_class))
    print('Accuracy Score :', accuracy_score(y_test, y_predict_class))
    print('ROC(Receiver Operating Characteristic) and AUC(Area Under Curve)', roc_auc_score(y_test, y_predict_class))
    print('Average Precision Score:', average_precision_score(y_test, y_predict_class))
    if svc.predict(tweetvec) == [1]:
        return "Positive"
    else:
        return "Negative"


    #Linear Support Vector Classifier Algorithm

def LinearSupportVectorClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec):
    print("Linear Support Vector Classifier")
    lsvc = LinearSVC(C=0.1)
    lsvc.fit(x_train_vec, y_train)
    y_predict_class = lsvc.predict(x_test_vec)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_predict_class))
    print('Accuracy Score :', accuracy_score(y_test, y_predict_class))
    print('ROC(Receiver Operating Characteristic) and AUC(Area Under Curve)', roc_auc_score(y_test, y_predict_class))
    print('Average Precision Score:', average_precision_score(y_test, y_predict_class))
    if lsvc.predict(tweetvec) == [1]:
        return "Positive"
    else:
        return "Negative"


# Decision Tree classification Algorithm

    #Decision Tree Classifier Algorithm

def DecisionTreeClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec):
    print("Decision Tree Classifier")
    dtc = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
    dtc.fit(x_train_vec, y_train)
    y_predict_class = dtc.predict(x_test_vec)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_predict_class))
    print('Accuracy Score :', accuracy_score(y_test, y_predict_class))
    print('ROC(Receiver Operating Characteristic) and AUC(Area Under Curve)', roc_auc_score(y_test, y_predict_class))
    print('Average Precision Score:', average_precision_score(y_test, y_predict_class))
    if dtc.predict(tweetvec) == [1]:
        return "Positive"
    else:
        return "Negative"


# Nearest Neighbors classification Algorithm

    #kNearest Neighbors Classifier Algorithm

def kNearestNeighborsClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec):
    print("K Nearest Neighbors")
    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(x_train_vec[0:8000], y_train[0:8000])
    y_predict_class = knn.predict(x_test_vec)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_predict_class))
    print('Accuracy Score :', accuracy_score(y_test, y_predict_class))
    print('ROC(Receiver Operating Characteristic) and AUC(Area Under Curve)', roc_auc_score(y_test, y_predict_class))
    print('Average Precision Score:', average_precision_score(y_test, y_predict_class))
    if knn.predict(tweetvec) == [1]:
        return "Positive"
    else:
        return "Negative"


    #Nearest Centroid Classifier Algorithm

def NearestCentroidAlgo(x_train_vec, y_train, x_test_vec, y_test, vec):
    print("Nearest Centroid Classifier")
    nc = NearestCentroid()
    nc.fit(x_train_vec, y_train)
    y_predict_class = nc.predict(x_test_vec)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_predict_class))
    print('Accuracy Score :', accuracy_score(y_test, y_predict_class))
    print('ROC(Receiver Operating Characteristic) and AUC(Area Under Curve)', roc_auc_score(y_test, y_predict_class))
    print('Average Precision Score:', average_precision_score(y_test, y_predict_class))
    if nc.predict(vec) == [1]:
        return "Positive"
    else:
        return "Negative"


# Ensemble classification Algorithm

    #Random Forest Classifier Algorithm

def RandomForestClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec):
    print("Random Forest Classifier")
    rfc = RandomForestClassifier(n_jobs=2, random_state=0)
    rfc.fit(x_train_vec, y_train)
    y_predict_class = rfc.predict(x_test_vec)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_predict_class))
    print('Accuracy Score :', accuracy_score(y_test, y_predict_class))
    print('ROC(Receiver Operating Characteristic) and AUC(Area Under Curve)', roc_auc_score(y_test, y_predict_class))
    print('Average Precision Score:', average_precision_score(y_test, y_predict_class))
    if rfc.predict(tweetvec) == [1]:
        return "Positive"
    else:
        return "Negative"