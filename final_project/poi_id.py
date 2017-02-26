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
 "total_stock_value", "long_term_incentive"]

# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']

### Task 3: Create new feature(s)


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

'''Changed sort_keys to False. Somehow getting very poor clf from True'''


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.5)


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


def pound_line():
    print "\n" + "#" * 60

def try_classifier(name, clf):
    pound_line()
    print " " * 20 + name + "\n"
    print "Features: ", features_list

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

##### Random Forest #####
# Seems to work best with specfic selected features
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
try_classifier("Random Forest", clf)

##### Extra Trees #####
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()
try_classifier("Extra Tree", clf)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
