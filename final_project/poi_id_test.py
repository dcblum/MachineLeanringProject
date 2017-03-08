#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi", "exercised_stock_options", "deferred_income",
 "expenses"]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']

### Task 3: Create new feature(s)

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


''' # normailize percent_to_poi and percent_from_poi and add together
normalize_feature('percent_to_poi', data_dict)
normalize_feature('percent_from_poi', data_dict)

# add normalized percent_to_poi and percent_from_poi to create email_poi_score
for person in data_dict:
    percent_to_norm = data_dict[person]['percent_to_poi']
    percent_from_norm = data_dict[person]['percent_from_poi']

    email_poi_score = percent_to_norm + percent_from_norm
    if email_poi_score >= 0:
        data_dict[person]['email_poi_score'] = email_poi_score


# Normalize features, DON'T normail poi (feature[0] is 'poi')
for feature in features_list[1:]:
    normalize_feature(feature, data_dict)
'''

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.33)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def test_prec_recall(name, clf_choice):
    precision_list = list()
    recall_list = list()
    for i in range(100):
        ### Extract features and labels from dataset for local testing
        data = featureFormat(data_dict, features_list, sort_keys = True)
        # Create labels and features
        labels, features = targetFeatureSplit(data)

        # transform into np.array for StratifiedShuffleSplit
        features = np.array(features)
        labels = np.array(labels)

        # Shuffle and split data into training/testing sets
        sss = StratifiedShuffleSplit()
        for train_index, test_index in sss.split(features, labels):
            features_train, features_test = features[train_index], features[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]


        # Create, fit, and predict classifier
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

    print "\n" + "#" * 60
    print " " * 20 + name + "\n"
    # print confusion_matrix(labels_test, labels_pred)
    print "Precision Mean: ", np.mean(precision_list)
    print "Recall Mean: ", np.mean(recall_list)
    print "STD_sum: ", np.std(precision_list) + np.std(recall_list)


##### Decision Tree #####
# Seems to work best with specfic selected features
from sklearn import tree
test_prec_recall("Decision Tree", tree.DecisionTreeClassifier())

##### Random Forest #####
# Does okay...
from sklearn.ensemble import RandomForestClassifier
test_prec_recall("Random Forest", RandomForestClassifier())

##### Extra Trees #####
# Current best if using all available features, but still just okay..
from sklearn.ensemble import ExtraTreesClassifier
test_prec_recall("Extra Tree", ExtraTreesClassifier())

##### SVMS #####
# So far linear, poly, and rfb SVMs are pretty bad at predicting pre-normalize
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipe = Pipeline(steps=[('minmaxer', MinMaxScaler()),  ('clf', SVC())])
#pipe.fit(features_train, features_test)


'''
hundred_test_prec_recall("SVM rbf", SVC(C=20, kernel='rbf'))
hundred_test_prec_recall("SVM poly", SVC(C=20, kernel='poly'))
'''


##### Naive Bayes #####
# Naive Bayes never predicts true positive, but can predict true negative.
from sklearn.naive_bayes import GaussianNB
test_prec_recall("Naive Bayes", GaussianNB())

###### Best clf appears to be Decison Tree;
###### precision and recall mean > .3
###### generally lowest sum of precision and recall standard deviations
clf = tree.DecisionTreeClassifier()

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
