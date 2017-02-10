#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC

clf = SVC(C = 10000, kernel='rbf')

# Reduce Training Set to 1% of total values
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), 's'

t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), 's'

#prettyPicture(clf, features_test, labels_test)
#plt.show()

from sklearn.metrics import accuracy_score

print "Acc. Score: ", accuracy_score(pred, labels_test)
print "Element 10: ", pred[10]
print "Element 26: ", pred[26]
print "Element 50: ", pred[50]

pred_chris = 0
for i in pred:
    if i == 1:
        pred_chris += 1
print "Pred Chris: ", pred_chris

#########################################################
