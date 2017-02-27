#Enron Project Fraud Classifer

###Goal: Use Machine Learning to Assess Fraud from Financial and Email Data 

Many companies keep datasets of thier financial and email data. These datasets are very large; it is extremely difficult to cipher through all of the data. Having a machine assist and find a potential, fraudulent person of interest (POI) would save both time and money.

The email data set tested is the Enron Corpus, which can be found at (https://www.cs.cmu.edu/~./enron/). Enron was one of the top energy companies in the US in the early 2000s. The company eventually filed Bankruptcy largely due to fraudulent cases of insider trading and accounting scandals. It is one of the most noteworthy cases of fraud in the 20th century. Because of the scale of the company and the size of the email database, the Enron Corpus is a prime dataset for finding clues of fraud via financials and email.

###Explore Data and Find Important Features

Initial exploration of the dataset revealed a couple outliers. Most entries were of people's names: 'SKILLING JEFFREY K' or 'LAY KENNETH L'; however both 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' were listed as if they were actual people that worked for Enron. Because both were not living beings, I decided to remove them from calculations. 

Features were hand picked from visual aide via https://public.tableau.com/profile/diego2420#!/vizhome/Udacity/UdacityDashboard. Further inspection of data could be made with already built-in-functions to find the best fit features.

Important Features picked for determining POI:
* exercised_stock_options (high values ~ POI)
* deferred_income (low values ~ POI)
* expenses (low values ~ not POI)

Three additional features were calculated but were found to be of minimal importance: percent_to_poi, percent_from_poi, and email_poi_score. The thought was to assign a score to each person signifying the capacity they were in contact with a POI. The score was calculated by summing a person's percent of emails _to_ or _from_ a POI. Again, this yielded no significant gain and was discarded as a feature.

###Classifiers

Data was tested with multiple classifiers consisting of individual trials of **raw** and **normalized** data. I noticed a very high variation when testing; some classifiers yielded high precision and recall scores, but when retested both scores occassionally dropped to zero. To counteract the high deviation of values, I wrote a few functions to generate 1000 classifiers of each type and compare their average scores.

Because SVM 'poly' took an extraordinant amount of time and never finished with raw data **only** a **normalized** data classifier was tested for SVM 'poly'

Output Data from Classifiers (normalized data only):
~~~
Note:  STD_sum is the sum of individual standard deviation scores (for simplification)
############################################################
                    DecisionTree

Precision Mean:  0.386631629482
Recall Mean:  0.381794300144
STD_sum:  0.426737950995
############################################################
                    RandomForest

Precision Mean:  0.381161507937
Recall Mean:  0.197606565657
STD_sum:  0.505336869258
############################################################
                    ExtraTree

Precision Mean:  0.4745251443
Recall Mean:  0.253108910534
STD_sum:  0.514709115944
############################################################
                    SVM 'rbf'

Precision Mean:  0.4215
Recall Mean:  0.0818132395382
STD_sum:  0.580025182133
############################################################
                    SVM 'poly'

Precision Mean:  0.024
Recall Mean:  0.00459642857143
STD_sum:  0.183943409806
############################################################
                    GaussianNB

Precision Mean:  0.522593650794
Recall Mean:  0.258691630592
STD_sum:  0.482226413296
############################################################
~~~

The classifier chosen as best is DecisionTree, followed somewhat closely by GaussianNB. DecisionTree maintains higher combined precision and recall scores while having relatively low standard deviations for both scores. Precision is the percent of people the classifier accurately labeled as a POI. Recall is the percent of actual POIs that were correctly classified. Although other classifiers have higher individual precision and recall scores (or individually having lower standard deviation) no other classifiers appear to be a better all-around compared to DecisionTree.

###Parameters

Multiple attempts were made to tune the SVM classifier. Two kernals were tested('rbf' and 'poly') along with varying C values from 1 - 1000. Trials to maximize SVM ceased when it appeared the classifier would never outclass either DecisionTree or GaussianNB. 

The DecisionTree paramenter 'criterion' was test for both 'gini' (default) and 'entropy'. Although close, 'entropy' introduced more deviation (results below) and was not used in the final calculation. 
~~~
############################################################
                    Decision Tree (criterion='entropy')

Precision Mean:  0.414058344433
Recall Mean:  0.383863888889
STD_sum:  0.46693555759
############################################################
~~~

###Validation

Validation is usually overcome by splitting the data into training and testing sets. A training size too large can overfit the classifier causing low testing results, whereas a trianing size too low can underfit the classfier, again, causing low testing results.  The Enron Corpus was split into both training (33%) and testing (67%) via function train_test_split with 'test_size = .33'. 

###Summary

The Enron Corpus is one of the largest datasets on fraud. Although the dataset isn't vast, a DecisionTree classifier appears to be a strong option in predicting fraud, followed closely by GaussianNB. Additional/different features for higher precision and recall scores are desired; the current scores appear mediocre but the classifier has potential to be a starting point for detecting fraud.  

####References

Feature Visualization:
https://public.tableau.com/profile/diego2420#!/vizhome/Udacity/UdacityDashboard

Forum Postings:
https://discussions.udacity.com/t/getting-started-with-final-project/170846

When to chose which machine learning classifier?
http://stackoverflow.com/questions/2595176/when-to-choose-which-machine-learning-classifier

