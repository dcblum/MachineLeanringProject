#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
all_features = ["poi", "salary", "to_messages", "deferral_payments",
"total_payments", "loan_advances", "bonus", "restricted_stock_deferred",
"deferred_income", "total_stock_value", "expenses", "from_poi_to_this_person",
"exercised_stock_options", "from_messages", "other", "from_this_person_to_poi",
"long_term_incentive", "shared_receipt_with_poi", "restricted_stock",
"director_fees", "email_address"]

current_max_features_list = ["poi", "exercised_stock_options", "deferred_income",
"total_stock_value", "long_term_incentive", "expenses"]

features_list = ["poi", "exercised_stock_options", "deferred_income",
"total_stock_value", "long_term_incentive", "expenses"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = train_test_split(
features, labels, test_size = .3, random_state = 42)

def pound_line():
    print "\n" + "#" * 60

def try_classifier(name, clf_choice):
    pound_line()
    print " " * 20 + name + "\n"

    print "Features: ", features_list

    clf = clf_choice

    t0 = time()
    clf.fit(features_train, labels_train)
    print "Training Time:", round(time()-t0, 3), 's'

    t0 = time()
    labels_pred = clf.predict(features_test)
    print "Predict Time:", round(time()-t0, 3), 's'

    print "Accuracy Score:", accuracy_score(labels_pred, labels_test)
    print "# of Features:", len(features[0])

    print
    print classification_report(labels_test, labels_pred)
    print confusion_matrix(labels_test, labels_pred)
    print

    print "Precision Score: ", precision_score(labels_test, labels_pred)
    print "Recall Score: ", recall_score(labels_test, labels_pred)


##### Decion Tree #####
# Seems to work best with specfic selected features
from sklearn import tree
try_classifier("Decision Tree", tree.DecisionTreeClassifier())

##### Random Forest #####
# Does okay...
from sklearn.ensemble import RandomForestClassifier
try_classifier("Random Forest", RandomForestClassifier())

##### Extra Trees #####
# Current best if using all available features, but still just okay..
from sklearn.ensemble import ExtraTreesClassifier
try_classifier("Extra Tree", ExtraTreesClassifier())

##### SVMS #####
# So far linear, poly, and rfb SVMs are pretty bad at predicting pre-normalize
# from sklearn.svm import SVC
# try_classifier("SVM rbf", SVC(C=20, kernel='rbf'))
# try_classifier("SVM poly", SVC(C=20, kernel='poly'))

##### Naive Bayes #####
# Naive Bayes never predicts true positive, but can predict true negative.
# from sklearn.naive_bayes import GaussianNB
# try_classifier("Naive Bayes", GaussianNB())
