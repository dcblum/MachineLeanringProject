# !/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from pprint import pprint
import numpy as np
import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

all_features = ['poi', 'salary', 'deferral_payments', 'total_payments','loan_advances',
                'bonus', 'restricted_stock_deferred', 'deferred_income','total_stock_value',
                'expenses', 'exercised_stock_options', 'other','long_term_incentive',
                'restricted_stock', 'director_fees', 'to_messages','from_poi_to_this_person',
                'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi',
                'email_poi_score']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Find number of data points and total POI
poi_list = list()

for name in data_dict:
    if data_dict[name]['poi']:
        poi_list.append(name)

print "Total Data Points:", len(data_dict)
print "Total POIs:", len(poi_list)

# Find percent NaN values in each feature
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype = np.float)
percent_nan_list = df.isnull().sum() / (df.isnull().sum() + df.notnull().sum())
print "\nPercent NaN Values:\n", percent_nan_list.sort_values(ascending = False)

### Task 2: Remove outliers
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']

### Task 3: Create new feature(s)

def normalize_feature(feature, data_dict):
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

# find percent emails sent to poi and percent from poi to this person
for person in data_dict:
    from_messages = data_dict[person]['from_messages']
    to_messages = data_dict[person]['to_messages']
    from_poi = data_dict[person]['from_poi_to_this_person']
    to_poi = data_dict[person]['from_this_person_to_poi']

    # Initialize all email_poi_score as 'NaN'
    data_dict[person]['email_poi_score'] = 'NaN'

    percent_to = float(to_poi) / float(from_messages)
    percent_from = float(from_poi) / float(to_messages)

    data_dict[person]['percent_to_poi'] = percent_to
    data_dict[person]['percent_from_poi'] = percent_from

# normailize percent_to_poi and percent_from_poi and add together
normalize_feature('percent_to_poi', data_dict)
normalize_feature('percent_from_poi', data_dict)

# add normalized percent_to_poi and percent_from_poi to create email_poi_score
for person in data_dict:
    percent_to_norm = data_dict[person]['percent_to_poi']
    percent_from_norm = data_dict[person]['percent_from_poi']

    email_poi_score = percent_to_norm + percent_from_norm
    if email_poi_score >= 0:
        data_dict[person]['email_poi_score'] = email_poi_score

# normalize 'email_poi_score'
normalize_feature('email_poi_score', data_dict)

### Store to my_dataset for easy export below.
my_dataset = data_dict

features_handpicked = ['poi', 'exercised_stock_options', 'deferred_income', 'expenses']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation
from operator import add
from heapq import nlargest

# Run loop to find how many times a feature occurs in the top 3
best_features = [0] * (len(all_features)-1)
for i in range(1000):
    # Create features and training labels
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = \
        cross_validation.train_test_split(features, labels, test_size=0.33)

    # Generate SelectKBest with k=3 features
    selector = SelectKBest(f_classif, k=3)
    selector.fit(features_train, labels_train)

    # Increase score of feature if it appears in the top 3
    best_features = selector.get_support().astype(int) + best_features

# Print the top 3 features scored by which features appeared most in top 3
features_kbest = ['poi']
for e in nlargest(3, best_features):
    for index in range(len(best_features)):
        if e == best_features[index]:
            top_feature = all_features[index+1]
            if top_feature not in features_kbest:
                features_kbest.append(top_feature)

print "Top 3 features:\n", features_kbest

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
from pprint import pprint



# Create test to find average precision and recall scores
def test_prec_recall(name, clf_choice, features_list):
    precision_list = list()
    recall_list = list()
    for i in range(1000):
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

    # F score is calculated via the mean precision and recall scores
    p_score = np.mean(precision_list)
    r_score = np.mean(recall_list)
    f_score = 2 * (p_score * r_score) / (p_score + r_score)

    print "\n" + "#" * 60
    print " " * 20 + name + "\n"
    print "Precision Mean Score: ", p_score
    print "Recall Mean Score: ", r_score
    print "F Score: ", f_score
    print "\n" + "#" * 60

'''from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier

pipeline_dt = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('clf', DecisionTreeClassifier(random_state = 49)),
])

param_grid_dt = {'pca__n_components': [2,3],
                 'imp__strategy': ['median', 'mean'],
                 'clf__min_samples_split': [2,3,5],
                 'clf__max_depth': [None,2,3],
                 }

cross_validator = StratifiedShuffleSplit(random_state = 0)


from sklearn.model_selection import GridSearchCV
gridCV_object_dt = GridSearchCV(estimator = pipeline_dt,
                                param_grid = param_grid_dt,
                                scoring = 'f1',
                                cv=cross_validator)

gridCV_object_dt = gridCV_object_dt.fit(features_train, labels_train)

test_prec_recall("Decision Tree: Hand-picked", gridCV_object_dt.best_estimator_, features_handpicked)

from sklearn.ensemble import RandomForestClassifier

pipeline_rfhp = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('clf', RandomForestClassifier(random_state = 49)),
])

param_grid_rfhp = {'pca__n_components': [2,3],
                 'imp__strategy': ['median', 'mean'],
                 'clf__n_estimators': [5,10,20],
                 'clf__min_samples_split': [2,5],
                 'clf__max_depth': [None,2,3],
                 }

cross_validator = StratifiedShuffleSplit(random_state = 0)

gridCV_object_rfhp = GridSearchCV(estimator = pipeline_rfhp,
                                param_grid = param_grid_rfhp,
                                scoring = 'f1',
                                cv=cross_validator)

gridCV_object_rfhp = gridCV_object_rfhp.fit(features_train, labels_train)

test_prec_recall("Random Forest: Hand-picked", gridCV_object_rfhp.best_estimator_, features_handpicked)

from sklearn.ensemble import ExtraTreesClassifier

pipeline_ethp = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('clf', ExtraTreesClassifier(random_state = 49)),
])

param_grid_ethp = {'pca__n_components': [2,3],
                 'imp__strategy': ['median', 'mean'],
                 'clf__n_estimators': [5,10,20],
                 'clf__min_samples_split': [2,5],
                 'clf__max_depth': [None,2,3],
                 }

cross_validator = StratifiedShuffleSplit(random_state = 0)

gridCV_object_ethp = GridSearchCV(estimator = pipeline_ethp,
                                param_grid = param_grid_ethp,
                                scoring = 'f1',
                                cv=cross_validator)

gridCV_object_ethp = gridCV_object_ethp.fit(features_train, labels_train)

test_prec_recall("Extra Tree: Hand-picked", gridCV_object_ethp.best_estimator_, features_handpicked)

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

pipeline_svchp = Pipeline([
    ('imp', Imputer()),
    ('minmaxscaler', MinMaxScaler()),
    ('pca', PCA()),
    ('clf', SVC()),
])

param_grid_svchp = {'imp__strategy': ['median', 'mean'],
                    'clf__C': [10,50,100],
                    'clf__kernel': ['rbf', 'poly'],
                    'pca__n_components': [2,3],
                   }

cross_validator = StratifiedShuffleSplit(random_state = 0)

gridCV_object_svchp = GridSearchCV(estimator = pipeline_svchp,
                                 param_grid = param_grid_svchp,
                                 scoring = 'f1',
                                 cv = cross_validator)

gridCV_object_svchp = gridCV_object_svchp.fit(features_train, labels_train)

test_prec_recall("SVM: Hand-picked", gridCV_object_svchp.best_estimator_, features_handpicked)

##### Naive Bayes #####

from sklearn.naive_bayes import GaussianNB

pipeline_nbhp = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('clf', GaussianNB())
])

param_grid_nbhp = {'pca__n_components': [1,2,3],
                 'imp__strategy': ['median', 'mean'],
                 }

cross_validator = StratifiedShuffleSplit(random_state = 0)

gridCV_object_nbhp = GridSearchCV(estimator = pipeline_nbhp,
                                 param_grid = param_grid_nbhp,
                                 scoring = 'f1',
                                 cv = cross_validator)

gridCV_object_svc = gridCV_object_nbhp.fit(features_train, labels_train)

test_prec_recall("Naive Bayes: Hand-picked", gridCV_object_nbhp.best_estimator_, features_handpicked)
'''


pipeline_dtsk = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('clf', DecisionTreeClassifier(random_state = 49)),
])

param_grid_dtsk = {'pca__n_components': [2,3],
                 'imp__strategy': ['median', 'mean'],
                 'clf__min_samples_split': [2,3,5],
                 'clf__max_depth': [None,2,3],
                 }

cross_validator = StratifiedShuffleSplit(random_state = 0)

gridCV_object_dtsk = GridSearchCV(estimator = pipeline_dtsk,
                                param_grid = param_grid_dtsk,
                                scoring = 'f1',
                                cv=cross_validator)

gridCV_object_dtsk = gridCV_object_dtsk.fit(features_train, labels_train)

test_prec_recall("Decision Tree: SelectKBest", gridCV_object_dtsk.best_estimator_, features_kbest)

'''
pipeline_rfsk = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('clf', RandomForestClassifier(random_state = 49)),
])

param_grid_rfsk = {'pca__n_components': [2,3],
                 'imp__strategy': ['median', 'mean'],
                 'clf__n_estimators': [5,10,20],
                 'clf__min_samples_split': [2,3,5],
                 'clf__max_depth': [None,2,3],
                 }

cross_validator = StratifiedShuffleSplit(random_state = 0)

gridCV_object_rfsk = GridSearchCV(estimator = pipeline_rfsk,
                                param_grid = param_grid_rfsk,
                                scoring = 'f1',
                                cv=cross_validator)

gridCV_object_rfsk = gridCV_object_rfsk.fit(features_train, labels_train)

test_prec_recall("Random Forest: SelectKBest", gridCV_object_rfsk.best_estimator_, features_kbest)

pipeline_etsk = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),

    ('clf', ExtraTreesClassifier(random_state = 49)),
])

param_grid_etsk = {'pca__n_components': [2,3],
                 'imp__strategy': ['median', 'mean'],
                 'clf__n_estimators': [5,10,20],
                 'clf__min_samples_split': [2,3,5],
                 'clf__max_depth': [None,2,3],
                 }

cross_validator = StratifiedShuffleSplit(random_state = 0)

gridCV_object_etsk = GridSearchCV(estimator = pipeline_etsk,
                                param_grid = param_grid_etsk,
                                scoring = 'f1',
                                cv=cross_validator)

gridCV_object_etsk = gridCV_object_etsk.fit(features_train, labels_train)

test_prec_recall("Extra Tree: SelectKBest", gridCV_object_etsk.best_estimator_, features_kbest)

pipeline_svcsk = Pipeline([
    ('imp', Imputer()),
    ('minmaxscaler', MinMaxScaler()),
    ('pca', PCA()),
    ('clf', SVC()),
])

param_grid_svcsk = {'imp__strategy': ['median', 'mean'],
                    'clf__C': [10,50,100],
                    'clf__kernel': ['rbf', 'poly'],
                    'pca__n_components': [2,3],
                   }

cross_validator = StratifiedShuffleSplit(random_state = 0)

gridCV_object_svcsk = GridSearchCV(estimator = pipeline_svcsk,
                                 param_grid = param_grid_svcsk,
                                 scoring = 'f1',
                                 cv = cross_validator)

gridCV_object_svcsk = gridCV_object_svcsk.fit(features_train, labels_train)

test_prec_recall("SVM: SelectKBest", gridCV_object_svcsk.best_estimator_, features_kbest)

pipeline_nbsk = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('clf', GaussianNB())
])

param_grid_nbsk = {'pca__n_components': [1,2,3],
                 'imp__strategy': ['median', 'mean'],
                 }

cross_validator = StratifiedShuffleSplit(random_state = 0)

gridCV_object_nbsk = GridSearchCV(estimator = pipeline_nbsk,
                                 param_grid = param_grid_nbsk,
                                 scoring = 'f1',
                                 cv = cross_validator)

gridCV_object_svcsk = gridCV_object_nbsk.fit(features_train, labels_train)

test_prec_recall("Naive Bayes: SelectKBest", gridCV_object_svcsk.best_estimator_, features_kbest)'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.33)

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(gridCV_object_dtsk.best_estimator_, my_dataset, features_kbest)
