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

current_max_features_list = ["poi", "exercised_stock_options",
"deferred_income", "total_stock_value", "long_term_incentive", "expenses"]

features_list = ["poi", "exercised_stock_options",
"deferred_income", "total_stock_value", "long_term_incentive", "expenses"]


# Remove Outliers
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']

''' # Find Poi with missing features
# Create dictionary of features with list of poi with associated missing feature
from pprint import pprint
missing_dict = dict()
for feature in features_list:
    missing_persons = []
    for person in data_dict:
        if data_dict[person]['poi'] == True:
            if data_dict[person][feature] == "NaN":
                missing_persons.append(person)
    if len(missing_persons) > 0:
        missing_dict[feature] = missing_persons

pprint(missing_dict)

# Finds poi with a SINGLE NaN in feature list
missing_list = list()
for feature in missing_dict:
    missing_list.append(missing_dict[feature])
missing_list = [item for sublist in missing_list for item in sublist]
from collections import Counter
missing_list = Counter(missing_list)

# Finds poi with NaN in ALL feature list
WARNING_LIST = list()
for person in missing_list:
    if missing_list[person] >= len(missing_dict):
        WARNING_LIST.append(person)
print "POI missing ALL features: ", WARNING_LIST
'''


''' # Normalize Data
from sklearn.preprocessing import normalize
import numpy as np

# Create list of arrays to normalize, then normalize
normalized_features_list = []
normalized_featuresbyname_dict = {}
# features_list[1:] to ignore if poi
for feature in features_list[1:]:
    feature_need_scale = []
    for person in data_dict:
        value = data_dict[person][feature]
        if value != 'NaN':
            feature_need_scale.append(value)

    normalized_features_list.append(normalize([feature_need_scale]))

print normalized_features_list
'''

'''
# Place normalized values back into data_dict
for feature in features_list[1:]:
        for person in range(len(data_dict)):
            number = data_dict[person][feature]
            if number != 'Nan':
                data_dict[person][feature] =
'''

data = featureFormat(data_dict, features_list, sort_keys = True)

print data
''' # Normalize Data after featureFormat
# Prepare normalization of data
from sklearn.preprocessing import normalize

# Create empty features list of lists ignoring poi
data_need_scale = [[] for i in range(len(features_list)-1)]
for person in data:
    for i in range(len(features_list)-1):
        data_need_scale[i].append(person[i+1])

# Scale each feature in the list (poi is not in list)
data_scaled = normalize(data_need_scale)

# Placed scaled features back into data (don't write into poi)
for person in range(len(data)):
    for feature in range(len(features_list)-1):
        data[person][feature+1] = data_scaled[feature][person]
'''
# Create labels and features
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

##### Decision Tree #####
# Seems to work best with specfic selected features
from sklearn import tree
try_classifier("Decision Tree", tree.DecisionTreeClassifier())

'''
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
from sklearn.svm import SVC
try_classifier("SVM rbf", SVC(C=20, kernel='rbf'))
try_classifier("SVM poly", SVC(C=20, kernel='poly'))

##### Naive Bayes #####
# Naive Bayes never predicts true positive, but can predict true negative.
from sklearn.naive_bayes import GaussianNB
try_classifier("Naive Bayes", GaussianNB())
'''
