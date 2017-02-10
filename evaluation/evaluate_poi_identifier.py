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
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn import tree
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

features_train, features_test, labels_train, labels_test = train_test_split(
features, labels, test_size = .3, random_state = 42)

clf = tree.DecisionTreeClassifier()

t0 = time()
clf.fit(features_train, labels_train)
print "Training Time:", round(time()-t0, 3), 's'

t0 = time()
pred = clf.predict(features_test)
print "Predict Time:", round(time()-t0, 3), 's'


from sklearn.metrics import accuracy_score
print "Accuracy Score:", accuracy_score(pred, labels_test)

print "# of Features:", len(features[0])

print classification_report(labels_test, pred)
print confusion_matrix(labels_test, pred)

print "Precision Score: ", precision_score(labels_test, pred)
print "Recall Score: ", recall_score(labels_test, pred)
