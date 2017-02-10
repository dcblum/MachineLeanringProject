#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here

    # Compute Residual Errors
    for i in range(len(predictions)):
        error = (predictions[i] - net_worths[i])**2
        cleaned_data.append((ages[i][0], net_worths[i][0], error[0]))

    # Remove top 10% of residual errors
    for i in range(int(len(cleaned_data)*.1)):
        cleaned_data.remove((max(cleaned_data, key = lambda x:x[2])))

    return cleaned_data
