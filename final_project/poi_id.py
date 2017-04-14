
# coding: utf-8

# # Machine Learning With Enron Corpus
#
# ## Goal: Use Machine Learning to Assess Fraud from Financial and Email Data
#
# Assessing fraud cases for any company is a tedious task that requires analyses across a vast amount of data. Fortunately most companies already have access to the data they need to combat fraud: emails and financials. However, these datasets are very large; it is extremely difficult to cipher through all of the data by hand. Having a machine assist and find a potential, fraudulent person of interest (POI) would save both time and money.
#
# The email data set tested is the Enron Corpus, which can be found at (https://www.cs.cmu.edu/~./enron/). Enron was one of the top energy companies in the US in the early 2000s. The company eventually filed Bankruptcy largely due to fraudulent cases of insider trading and accounting scandals. It is one of the most noteworthy cases of fraud in the 20th century. Because of the scale of the company and the size of the email database, the Enron Corpus is a prime dataset for finding clues of fraud via financials and email.

# In[1]:

# !/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from pprint import pprint
import numpy as np


# ## Explore Data and Find Important Features

# In[2]:

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

all_features = ['poi', 'loan_advances', 'director_fees',
'restricted_stock_deferred', 'deferral_payments', 'deferred_income',
'long_term_incentive', 'bonus', 'from_poi_to_this_person',
'shared_receipt_with_poi', 'to_messages', 'from_this_person_to_poi',
'to_messages', 'from_this_person_to_poi', 'from_messages', 'other',
'expenses', 'salary', 'exercised_stock_options','restricted_stock',
'total_payments', 'total_stock_value', 'email_address']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[3]:

# Find number of data points and total POI
poi_list = list()

for name in data_dict:
    if data_dict[name]['poi']:
        poi_list.append(name)

print "Total Data Points:", len(data_dict)
print "Total POIs:", len(poi_list)


# In[4]:

import pandas as pd
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype = np.float)
percent_nan_list = df.isnull().sum() / (df.isnull().sum() + df.notnull().sum())
print "\nPercent NaN Values:\n", percent_nan_list.sort_values(ascending = False)


# Initial exploration of the dataset revealed 146 data points with 18 POIs. Every data point consisted of features as mentioned above such as: salary, total_payment, to_messages, and expenses.
#
# Most features in the Enron Corpus contain NaN values, and these NaN values make up greater than 40% of most features. NaN values represent a lack of information and weaken the overall influence and accuracy of a feature when testing for fraud in the database. There are multiple methods to handle NaN values; in this project NaN values were changed to be either the mean or median by use of a GridCV object and the Imputer function.
#
# Features with higher than 80% were not used in calculations.

# In[5]:

features_list = ['poi', 'salary', 'total_payments','bonus', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'to_messages',
'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
'shared_receipt_with_poi', 'email_poi_score']


# Digging into the dataset revealed not all entries were people's names: 'SKILLING JEFFREY K' or 'LAY KENNETH L'; however both 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' were listed as if they were actual people that worked for Enron. Because both were not living beings, I decided to remove them from calculations.

# In[6]:

### Task 2: Remove outliers
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']


# After removing the outliers, three additional features were calculated but were later found to be of minimal importance: percent_to_poi, percent_from_poi, and email_poi_score.
#
# The thought was to assign a score to each person signifying the capacity they were in contact with a POI. The score was calculated by summing a person's percent of emails _to_ or _from_ a POI.

# In[7]:

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


# In[8]:

### Store to my_dataset for easy export below.
my_dataset = data_dict


# ##### Hand-picked Feature Selection
#
# Feature selection was initially hand-picked from visual aide via https://public.tableau.com/profile/diego2420#!/vizhome/Udacity/UdacityDashboard. Features were chosen based on visual clumping of POIs and non-POIs. The number of features were chosen somewhat arbitrarily; only features that appeared to have a strong visual clumping were chosen.
#
# Hand-picked features for determining POI:
# * exercised_stock_options (high values ~ POI)
# * deferred_income (low values ~ POI)
# * expenses (low values ~ not POI)

# In[9]:

features_handpicked = ['poi', 'exercised_stock_options', 'deferred_income',
'expenses']


# ##### SelectKBest Feature Selection
#
# Features were also chosen using SelectKBest. Because top features selected from SelectKBest can change depending on the randomness of training and testing the data, a tally was taken to determine which features appear in the top 3 features the most over 1000 trials. The idea behind only choosing the top 3 is only 3 features were chosen for the hand-picked test.
#
# SelectKBest features for determining POI:
# * exercised_stock_option
# * total_stock_value
# * bonus
#

# In[10]:

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation
from operator import add
from heapq import nlargest

# Run loop to find how many times a feature occurs in the top 3
best_features = [0] * (len(features_list)-1)
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


print "In top 3:\n", best_features
# Print the top 3 features scored by which features appeared most in top 3
features_kbest = ['poi']
for e in nlargest(3, best_features):
    for index in range(len(best_features)):
        if e == best_features[index]:
            top_feature = features_list[index+1]
            if top_feature not in features_kbest:
                features_kbest.append(top_feature)

print "\nTop 3 features:\n", features_kbest


# ## Classifiers
#
# To test for optimal training from the data multiple classifiers are used:
# * Decision Tree
# * Random Forest
# * Extra Trees
# * SVMs
# * GaussianNB
#
# Classifiers were tested for high scores in precision, recall, and f1. Precision is the percent of people the classifier accurately labeled as a POI. Recall is the percent of actual POIs that were correctly classified. The F1 score relates both precion and recall. F1 scores are calculated by this equation:
#
# $$ F1 = 2 * (precision * recall)  /  (precision + recall) $$
#
# Only classifiers with precsion and recall scores greater than or equal to .33 will be considered for the best overall classifier. Any of those best classifiers will then be ranked by highest f1 score.
#
# During initial testings I noticed a very high variation of all scores; classifiers that yielded high precision and recall scores may have low scores on the next test. To counteract the high deviation of values, I wrote a test function to generate 1000 classifiers of each type and compare their average scores.
#

# ## Classifiers (Features List)

# The features list contains all features with percent of NaN values greater than 80%.

# ##### Classifiers: Decision Tree (Features List)

# In[11]:

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[12]:

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif

pipeline_dt = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', DecisionTreeClassifier()),
])

param_grid_dt = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },

]

cross_validator = StratifiedShuffleSplit()


from sklearn.model_selection import GridSearchCV
gridCV_object_dt = GridSearchCV(estimator = pipeline_dt,
                                param_grid = param_grid_dt,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_dt.fit(features, labels)

# get the best estimator
pipeline_clf_dt_af = gridCV_object_dt.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_dt_af, my_dataset, features_list)


# The DecisionTree Classifier yielded results of:
# * Precision: .266
# * Recall: .269
# * F1: .267
#
# The features list with the DecisionTree Classifier does not have high enough results.

# ##### Classifiers: Random Forest (Features List)

# In[13]:

from sklearn.ensemble import RandomForestClassifier

pipeline_rf = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier()),
])

param_grid_rf = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
]

cross_validator = StratifiedShuffleSplit(random_state = 0)


from sklearn.model_selection import GridSearchCV
gridCV_object_rf = GridSearchCV(estimator = pipeline_rf,
                                param_grid = param_grid_rf,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_rf.fit(features, labels)

# get the best estimator
pipeline_clf_rf_af = gridCV_object_rf.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_rf_af, my_dataset, features_list)


# The RandomForest Classifier yielded results of:
# * Precision: .556
# * Recall: .084
# * F1: .146
#
# The features list with the RandomForest Classifier does not have high enough results.

# ##### Classifiers: Extra Trees (Features List)

# In[14]:

from sklearn.ensemble import ExtraTreesClassifier

pipeline_et = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', ExtraTreesClassifier()),
])

param_grid_et = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
]

cross_validator = StratifiedShuffleSplit(random_state = 0)


from sklearn.model_selection import GridSearchCV
gridCV_object_et = GridSearchCV(estimator = pipeline_et,
                                param_grid = param_grid_et,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_et.fit(features, labels)

# get the best estimator
pipeline_clf_et_af = gridCV_object_et.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_et_af, my_dataset, features_list)


# The ExtraTrees Classifier yielded results of:
# * Precision: .417
# * Recall: .128
# * F1: .200
#
# The features list with the ExtraTrees Classifier does not have high enough results.

# ##### Classifiers: SVMs (Features List)

# In[15]:

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

pipeline_svc = Pipeline([
    ('imp', Imputer()),
    ('minmaxscaler', MinMaxScaler()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', SVC()),
])

param_grid_svc = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__C': [10,50,100],
        'clf__kernel': ['rbf', 'poly'],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [4],
        'skb__k': [1,2,3,4],
        'clf__C': [10,50,100],
        'clf__kernel': ['rbf', 'poly'],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [5],
        'skb__k': [1,2,3,4,5],
        'clf__C': [10,50,100],
        'clf__kernel': ['rbf', 'poly'],
    }
]

cross_validator = StratifiedShuffleSplit()

gridCV_object_svc = GridSearchCV(estimator = pipeline_svc,
                                 param_grid = param_grid_svc,
                                 scoring = 'f1',
                                 cv = cross_validator)

# fit the data
gridCV_object_svc.fit(features_train, labels_train)


# get the best estimator
pipeline_clf_svc_af = gridCV_object_svc.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_svc_af, my_dataset, features_list)


# The SVM Classifier yielded results of:
# * Precision: .406
# * Recall: .045
# * F1: .080
#
# The features list with the SVM Classifier does not have high enough results.

# ##### Classifiers: Naive Bayes (Features List)

# In[16]:

from sklearn.naive_bayes import GaussianNB

pipeline_nb = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier()),
])

param_grid_nb = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [4],
        'skb__k': [1,2,3,4],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [5],
        'skb__k': [1,2,3,4,5],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [6],
        'skb__k': [1,2,3,4,5,6],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    }
]

cross_validator = StratifiedShuffleSplit(random_state = 0)


from sklearn.model_selection import GridSearchCV
gridCV_object_nb = GridSearchCV(estimator = pipeline_nb,
                                param_grid = param_grid_nb,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_nb.fit(features, labels)

# get the best estimator
pipeline_clf_nb_af = gridCV_object_nb.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_nb_af, my_dataset, features_list)


# The NaiveBayes Classifier yielded results of:
# * Precision: .415
# * Recall: .114
# * F1: .179
#
# The features list with the NaiveBayes Classifier does not have high enough results.

# ### Classifiers Summary (Features List)
#
# None of the classifiers using Features List performed well enough. It is likely using fewer features may result in higher scores; this will be tested further using hand-picked features and features selected using SelectKBest.

# ## Classifiers (Hand-Picked)

# ##### Classifiers: Decision Tree (Hand-Picked)

# In[17]:

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_handpicked, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[18]:

pipeline_dt = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', DecisionTreeClassifier()),
])

param_grid_dt = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    }
]

cross_validator = StratifiedShuffleSplit()


from sklearn.model_selection import GridSearchCV
gridCV_object_dt = GridSearchCV(estimator = pipeline_dt,
                                param_grid = param_grid_dt,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_dt.fit(features, labels)

# get the best estimator
pipeline_clf_dt_hp = gridCV_object_dt.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_dt_hp, my_dataset, features_handpicked)


# The DecisionTree Classifier yielded results of:
# * Precision: .351
# * Recall: .324
# * F1: .337
#
# The hand-picked features with the DecisionTree Classifier does not have high enough results. However, the results are _very_ close to passing.

# ##### Classifiers: Random Forest (Hand-Picked)

# In[19]:

pipeline_rf = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier()),
])

param_grid_rf = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    }
]

cross_validator = StratifiedShuffleSplit(random_state = 0)


gridCV_object_rf = GridSearchCV(estimator = pipeline_rf,
                                param_grid = param_grid_rf,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_rf.fit(features, labels)

# get the best estimator
pipeline_clf_rf_hp = gridCV_object_rf.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_rf_hp, my_dataset, features_handpicked)


# The RandomTrees Classifier yielded results of:
# * Precision: .557
# * Recall: .241
# * F1: .336
#
# The hand-picked features with the RandomTrees Classifier does not have high enough results.

# ##### Classifiers: Extra Trees (Hand-Picked)

# In[20]:

pipeline_et = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', ExtraTreesClassifier()),
])

param_grid_et = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    }
]

cross_validator = StratifiedShuffleSplit(random_state = 0)

gridCV_object_etsk = GridSearchCV(estimator = pipeline_et,
                                param_grid = param_grid_et,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_et.fit(features, labels)

# get the best estimator
pipeline_clf_et_hp = gridCV_object_et.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_et_hp, my_dataset, features_handpicked)


# The ExtraTrees Classifier yielded results of:
# * Precision: .544
# * Recall: .205
# * F1: .298
#
# The hand-picked features with the ExtraTrees Classifier does not have high enough results.

# ##### Classifiers: SVMs (Hand-Picked)

# In[21]:

pipeline_svc = Pipeline([
    ('imp', Imputer()),
    ('minmaxscaler', MinMaxScaler()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', SVC()),
])

param_grid_svc = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__C': [10,50,100],
        'clf__kernel': ['rbf', 'poly'],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__C': [10,50,100],
        'clf__kernel': ['rbf', 'poly'],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__C': [10,50,100],
        'clf__kernel': ['rbf', 'poly'],
    }
]

cross_validator = StratifiedShuffleSplit()

gridCV_object_svc = GridSearchCV(estimator = pipeline_svc,
                                 param_grid = param_grid_svc,
                                 scoring = 'f1',
                                 cv = cross_validator)

# fit the data
gridCV_object_svc.fit(features_train, labels_train)


# get the best estimator
pipeline_clf_svc_hp = gridCV_object_svc.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_svc_hp, my_dataset, features_handpicked)


# The SVM Classifier yielded results of:
# * Precision: .730
# * Recall: .119
# * F1: .205
#
# The hand-picked features with the SVM Classifier does not have high enough results.

# ##### Classifiers: Naive Bayes (Hand-Picked)

# In[22]:

pipeline_nb = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier()),
])

param_grid_nb = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    }
]

cross_validator = StratifiedShuffleSplit(random_state = 0)


from sklearn.model_selection import GridSearchCV
gridCV_object_nb = GridSearchCV(estimator = pipeline_nb,
                                param_grid = param_grid_nb,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_nb.fit(features, labels)

# get the best estimator
pipeline_clf_nb_hp = gridCV_object_nb.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_nb_hp, my_dataset, features_handpicked)


# The NaiveBayes Classifier yielded results of:
# * Precision: .553
# * Recall: .243
# * F1: .337
#
# The hand-picked features with the NaiveBayes Classifier does not have high enough results.

# ### Classifiers: Hand-picked Summary
#
# None of the classifiers using the hand-picked features performed well enough. However the DecisionTree Classifier yields results that are extremtly close to the .33 threshold.

# ## Classifiers (SelectKBest Features)

# ##### Classifiers: Decision Tree (SelectKBest)

# In[30]:

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_kbest, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[31]:

pipeline_dt = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', DecisionTreeClassifier()),
])

param_grid_dt = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    }
]

cross_validator = StratifiedShuffleSplit()


from sklearn.model_selection import GridSearchCV
gridCV_object_dt = GridSearchCV(estimator = pipeline_dt,
                                param_grid = param_grid_dt,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_dt.fit(features, labels)

# get the best estimator
pipeline_clf_dt_sk = gridCV_object_dt.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_dt_sk, my_dataset, features_kbest)


# The DecisionTree Classifier yielded results of:
# * Precision: .596
# * Recall: .456
# * F1: .517
#
# The selectkbest features with the DecisionTree Classifier far exceeded result expectations.

# ##### Classifiers: Random Forest (SelectKBest)

# In[25]:

pipeline_rf = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier()),
])

param_grid_rf = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    }
]

cross_validator = StratifiedShuffleSplit(random_state = 0)


gridCV_object_rf = GridSearchCV(estimator = pipeline_rf,
                                param_grid = param_grid_rf,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_rf.fit(features, labels)

# get the best estimator
pipeline_clf_rf_sk = gridCV_object_rf.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_rf_sk, my_dataset, features_kbest)


# The RandomTrees Classifier yielded results of:
# * Precision: .369
# * Recall: .252
# * F1: .299
#
# The selectkbest features with the RandomTrees Classifier does not have high enough results.

# ##### Classifiers: Extra Trees (SelectKBest)

# In[26]:

pipeline_et = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', ExtraTreesClassifier()),
])

param_grid_et = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    }
]

cross_validator = StratifiedShuffleSplit(random_state = 0)

gridCV_object_etsk = GridSearchCV(estimator = pipeline_et,
                                param_grid = param_grid_et,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_et.fit(features, labels)

# get the best estimator
pipeline_clf_et_sk = gridCV_object_et.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_et_sk, my_dataset, features_kbest)


# The ExtraTrees Classifier yielded results of:
# * Precision: .594
# * Recall: .252
# * F1: .354
#
# The selectkbest features with the ExtraTrees Classifier does not have high enough results.

# ##### Classifiers: SVMs (SelectKBest)

# In[27]:

pipeline_svc = Pipeline([
    ('imp', Imputer()),
    ('minmaxscaler', MinMaxScaler()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', SVC()),
])

param_grid_svc = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__C': [10,50,100],
        'clf__kernel': ['rbf', 'poly'],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__C': [10,50,100],
        'clf__kernel': ['rbf', 'poly'],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__C': [10,50,100],
        'clf__kernel': ['rbf', 'poly'],
    }
]

cross_validator = StratifiedShuffleSplit()

gridCV_object_svc = GridSearchCV(estimator = pipeline_svc,
                                 param_grid = param_grid_svc,
                                 scoring = 'f1',
                                 cv = cross_validator)

# fit the data
gridCV_object_svc.fit(features_train, labels_train)


# get the best estimator
pipeline_clf_svc_sk = gridCV_object_svc.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_svc_sk, my_dataset, features_kbest)


# The SVM Classifier yielded results of:
# * Precision: .778
# * Recall: .132
# * F1: .225
#
# The selectkbest features with the SVM Classifier does not have high enough results.

# ##### Classifiers: Naive Bayes (SelectKBest)

# In[28]:

pipeline_nb = Pipeline([
    ('imp', Imputer()),
    ('pca', PCA()),
    ('skb', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier()),
])

param_grid_nb = [
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [1],
        'skb__k': [1],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [2],
        'skb__k': [1,2],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    },
    {
        'imp__strategy': ['median', 'mean'],
        'pca__n_components': [3],
        'skb__k': [1,2,3],
        'clf__min_samples_split': [2,3,5],
        'clf__max_depth': [None,2,3],
    }
]

cross_validator = StratifiedShuffleSplit(random_state = 0)


from sklearn.model_selection import GridSearchCV
gridCV_object_nb = GridSearchCV(estimator = pipeline_nb,
                                param_grid = param_grid_nb,
                                scoring = 'f1',
                                cv=cross_validator)

# fit the data
gridCV_object_nb.fit(features, labels)

# get the best estimator
pipeline_clf_nb_sk = gridCV_object_nb.best_estimator_

# test results
from tester import test_classifier
test_classifier(pipeline_clf_nb_sk, my_dataset, features_kbest)


# The NaiveBayes Classifier yielded results of:
# * Precision: .457
# * Recall: .239
# * F1: .313
#
# The selectkbest features with the NaiveBayes Classifier does not have high enough results.

# ### Classifiers: SelectKBest Summary
#
# The Decision Tree classifier completely exceeds all expections and is the clear winner of the selectkbest feature selection. The **Decision Tree** classifier with **selectkbest** attained scores of **precision = .596**, **recall = .456** and **F score = .517**.

# ### Classifiers: Features List vs. Hand-picked vs. SelectKBest Summary
#
# The features chosen as best across features list, hand-picked, and selectkbest is the features chosen with selectkbest. Again, the selectkbest features are:
# * exercised_stock_option
# * total_stock_value
# * bonus
#
# The Classifier that offered the greatest results is the DecisionTree Classifier.
#
# Combing both the selectkbest features and the DecisionTree Classifier results yielded an astounding:
# * Precision: .596
# * Recall: .456
# * F1: .517

# ### Validation and Parameter Tuning
#
# ##### Validation
# Data validation is the series of steps it takes to make sure the data is clean and useful. Dirty data (being mislabeled or unaccurate) leads to inconclusive results no matter the outcome.
#
# Validation in machine learning is first overcome by exploring the dataset and handling dirty data through means of cleaning it up or removing it entirely. This was achieved in the Enron Corpus by removing non-name entries and later by the Imputer function to change NaN values to either mean or median values (whichever lead to higher f1 scores). Another validation technique used was splitting the data into training and testing sets. A training size too large can overfit the classifier causing low testing results, whereas a trianing size too low can underfit the classfier, again, causing low testing results.
#
#
# ##### Parameter Tuning
# As discussed earlier, parameters of each classifier were tuned using both a pipeline and creating a gridCV object. The gridCV object tests multiple lists of parameters and returns the parameters that maximize a scoring function. In this case all parameters were tuned to maximize F1 scores.
#
# ### Summary
#
# The Enron Corpus is one of the largest datasets on fraud. Although the dataset isn't vast, a Decision Tree classifier appears to be a strong option in predicting fraud.
#
# The most important features to analyze for attempting fraud are most likely:
# * exercised stock options
# * total stock
# * bonus
#
# The afformetioned features combined with a Decision Tree classifier yield precision, recall, and f1 scores close to .4. Additional/different features for higher scores are desired; the current scores appear mediocre but the classifier has potential to be a starting point for detecting fraud.
#
# #### References
#
# Feature Visualization:
# https://public.tableau.com/profile/diego2420#!/vizhome/Udacity/UdacityDashboard
#
# Forum Postings:
# https://discussions.udacity.com/t/getting-started-with-final-project/170846
#
# When to chose which machine learning classifier:
# http://stackoverflow.com/questions/2595176/when-to-choose-which-machine-learning-classifier
#
# GridCV and Pipeline testing:
# https://discussions.udacity.com/t/webcast-builidng-models-with-gridsearchcv-and-pipelines-thursday-11-feb-2015-at-6pm-pacific-time/47412

# In[33]:

dump_classifier_and_data(pipeline_clf_dt_sk, my_dataset, features_kbest)


# In[ ]:
