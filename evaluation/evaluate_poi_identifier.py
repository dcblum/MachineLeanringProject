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
import numpy as np

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

### Features: excluding 'email_address'
all_features = ['poi', 'salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages',
'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
'shared_receipt_with_poi']

current_max_features_list = ["poi", "exercised_stock_options",
"deferred_income", "total_stock_value", "expenses"]

features_list = ["poi", "exercised_stock_options", "deferred_income",
 "expenses"]


# Remove Outliers
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']


# Create normalize function
def normalize_feature(feature, data_dict):
    '''normalize_feature([string], [dictionary]):
    Normalizes all numerical values of key 'string' in a dictionary'''
    # features_list[1:] to ignore poi feature
    # initialize high and low value for normalization function
    value_high = None
    value_low = None

    # loop through persons to find high and low values for features
    for person in data_dict:
        value = data_dict[person][feature]
        if value != 'NaN':
            # If first value in feature then assign value to variables
            if value_low == None:
                value_high = value
                value_low = value
            # look to see if value is higher or lower
            if value > value_high:
                value_high = value
            elif value < value_low:
                value_low = value

    # loop to assign normalization value
    for person in data_dict:
        value = float(data_dict[person][feature])
        # if value exists between high and low
        if (value_high >= value) and (value_low <= value):
            # if denominator isn't zero
            if value_high != value_low:
                value_norm = (value - value_low) / (value_high - value_low)
                data_dict[person][feature] = value_norm


# Create feature: email_poi_score
# find percent emails sent to poi and percent from poi to this person
for person in data_dict:
    from_messages = data_dict[person]['from_messages']
    to_messages = data_dict[person]['to_messages']
    from_poi = data_dict[person]['from_poi_to_this_person']
    to_poi = data_dict[person]['from_this_person_to_poi']

    data_dict[person]['email_poi_score'] = 'NaN'


    percent_to = float(to_poi) / float(from_messages)
    percent_from = float(from_poi) / float(to_messages)

    data_dict[person]['percent_to_poi'] = percent_to
    data_dict[person]['percent_from_poi'] = percent_from

# normailize each and add together
normalize_feature('percent_to_poi', data_dict)
normalize_feature('percent_from_poi', data_dict)

# add normalized percent_to_poi and percent_from_poi to create email_poi_score
for person in data_dict:
    percent_to_norm = data_dict[person]['percent_to_poi']
    percent_from_norm = data_dict[person]['percent_from_poi']

    email_poi_score = percent_to_norm + percent_from_norm
    if email_poi_score >= 0:
        data_dict[person]['email_poi_score'] = email_poi_score

'''
# Find Poi with missing features
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

#pprint(missing_dict)

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
#print "POI missing ALL features: ", WARNING_LIST
'''


# Normalize All Data from 0 to 1
# Find all values in a feature, calculate normalization, then reassign value
for feature in features_list[1:]:
    normalize_feature(feature, data_dict)



### your code goes here
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

'''# Combined into function hundred_test_prec_recall

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)

# Create labels and features
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(
features, labels, test_size = .3)
'''

def pound_line():
    print "\n" + "#" * 60

def try_classifier(name, clf_choice):
    pound_line()
    print " " * 20 + name + "\n"
    #print "Features: ", features_list

    clf = clf_choice
    t0 = time()
    clf.fit(features_train, labels_train)
    #print "Training Time:", round(time()-t0, 3), 's'

    t0 = time()
    labels_pred = clf.predict(features_test)
    #print "Predict Time:", round(time()-t0, 3), 's'
    #print "Accuracy Score:", accuracy_score(labels_pred, labels_test)
    #print "# of Features:", len(features[0])

    print
    #print classification_report(labels_test, labels_pred)
    #print confusion_matrix(labels_test, labels_pred)
    print

    print "Precision Score: ", precision_score(labels_test, labels_pred)
    print "Recall Score: ", recall_score(labels_test, labels_pred)

def hundred_test_prec_recall(name, clf_choice):
    precision_list = list()
    recall_list = list()
    for i in range(1000):
        ### Extract features and labels from dataset for local testing
        data = featureFormat(data_dict, features_list, sort_keys = True)
        # Create labels and features
        labels, features = targetFeatureSplit(data)
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size = .33)

        clf = clf_choice
        clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)

        try:
            precision = precision_score(labels_test, labels_pred)
            recall = recall_score(labels_test, labels_pred)
            precision_list.append(precision)
            recall_list.append(recall)
        except:
            pass

    pound_line()
    print " " * 20 + name + "\n"
    print confusion_matrix(labels_test, labels_pred)
    print "Precision Mean: ", np.mean(precision_list)
    print "Recall Mean: ", np.mean(recall_list)
    print "STD_sum: ", np.std(precision_list) + np.std(recall_list)

for person in data_dict:
    if data_dict[person]['poi']:
        print person

'''
##### Decision Tree #####
# Seems to work best with specfic selected features
from sklearn import tree
hundred_test_prec_recall("Decision Tree", tree.DecisionTreeClassifier())

##### Random Forest #####
# Does okay...
from sklearn.ensemble import RandomForestClassifier
hundred_test_prec_recall("Random Forest", RandomForestClassifier())

##### Extra Trees #####
# Current best if using all available features, but still just okay..
from sklearn.ensemble import ExtraTreesClassifier
hundred_test_prec_recall("Extra Tree", ExtraTreesClassifier())

##### SVMS #####
# So far linear, poly, and rfb SVMs are pretty bad at predicting pre-normalize
from sklearn.svm import SVC
hundred_test_prec_recall("SVM rbf", SVC(C=20, kernel='rbf'))
hundred_test_prec_recall("SVM poly", SVC(C=20, kernel='poly'))

##### Naive Bayes #####
# Naive Bayes never predicts true positive, but can predict true negative.
from sklearn.naive_bayes import GaussianNB
hundred_test_prec_recall("Naive Bayes", GaussianNB())

pound_line()
'''
