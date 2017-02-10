#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print "People: ", len(enron_data)
print "Features: ", len(enron_data.values()[0])
print "# POI: ", sum([x['poi'] for x in enron_data.values()])
print enron_data["PRENTICE JAMES"]['total_stock_value']
print enron_data["COLWELL WESLEY"]['from_this_person_to_poi']
print enron_data["SKILLING JEFFREY K"]['exercised_stock_options']
print enron_data["SKILLING JEFFREY K"]['total_payments']
print enron_data["LAY KENNETH L"]['total_payments']
print enron_data["FASTOW ANDREW S"]['total_payments']
print "# Have Salary: ", sum([x['salary']!='NaN' for x in enron_data.values()])
print "# E-Address: ", sum([x['email_address']!='NaN' for x in enron_data.values()])
print "# NaN t_pay: ", sum([x['total_payments']=='NaN' for x in enron_data.values()])
print "# NaN POI t_pay: ",
sum([x['total_payments']=='NaN' and x['poi'] == True for x in enron_data.values()])
