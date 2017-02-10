#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL')
data = featureFormat(data_dict, features)

### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

for name in data_dict:
    if (data_dict[name]['salary'] >= 1000000) and data_dict[name]['salary']!='NaN':
        if data_dict[name]['bonus'] >= 5000000:
            print 'Name: ', name
            print 'Salary: ', data_dict[name]['salary']
            print 'Bonus: ', data_dict[name]['bonus']

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
