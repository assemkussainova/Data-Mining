import sys
import math
import time
import pandas as pd

# Reading data from selected file
def read_data(f):
    with open(f, 'r') as f:
        string = f.readline()
        while string:
            string = string.replace(" ", "") # remove spaces
            string = string.replace(";", "") # remove ';' char
            if string.startswith('T'):
                columns = string[2:len(string)].split(',') # define column names for dataframes starting after "T:"
                train = pd.DataFrame(columns = columns)
                test = pd.DataFrame(columns = columns)
            elif string.startswith('A'):
                train = train.append(pd.Series(string[2:len(string)].split(','), index = columns), ignore_index = True) # add string to train data starting from "A:"
            elif string.startswith('B'):
                test = test.append(pd.Series(string[2:len(string)].split(','), index = columns), ignore_index = True) # add string to train data starting from "B:"
            string = f.readline()

    feat, out = train.columns[:-1], train.columns[-1] # define features and target names
    return train, test, feat, out


# Identify best rule for the algorithm
def best_predictor(df, f, o):
    dictionary = {}
    for feature in f:
        count = 0
        for val in df[feature].unique():
            count += df[df[feature] == val][o].value_counts().max() # count the maximum number of feature-target combinations and assign to the features
        dictionary[feature] = count

    pred_name = max(dictionary, key = dictionary.get) # get best feature with highest accuracy provided
    pred_score = max(dictionary.values()) # get the corresponding value for best feature
    return pred_name, pred_score


# Print results of the algorithm in a given format
def print_results(start, r, r_score, train_dat, test_dat, out):
    for_output = {}
    print(f"{r}: ({r_score}/{len(train_dat)})")

    for val in train_dat[r].unique():
        freq = train_dat[train_dat[r] == val][out].value_counts()
        print(f"  {val} -> {freq.index[0]} ({freq[0]}/{freq.sum()})") # print all the values from chosen feature with their scores
        for_output[val] = freq.index[0]

    targets = train_dat[out].value_counts()
    print(f"  [ZeroRule] -> {targets.index[0]} ({targets[0]}/{targets.sum()})")
    max_zero = targets.index[0] # zero rule target answer

    seconds = time.time() - start # calculate time for training system
    millisec = math.modf(seconds)[0] * 1000
    minutes = int((millisec / (1000 * 60)) % 60)
    microsec = math.modf(millisec)[0] * 1000
    nanosec = math.modf(microsec)[0] * 1000
    print(f"Training time: {int(minutes)} minutes {int(seconds)} seconds {int(millisec)} milliseconds {int(microsec)} microseconds {int(nanosec)} nanoseconds \n")

    # run algorithm on testing data
    start = time.time()
    test_pred = 0
    for val in test_dat[r].unique():
        test_pred += sum(test_dat[test_dat[r] == val][out] == for_output.get(val, max_zero))

    print(f"Accuracy on test data: {test_pred}/{len(test_dat)}\n") # print accuracy on testing data
    seconds = time.time() - start # calculate time for testing system
    millisec = math.modf(seconds)[0] * 1000
    minutes = int((millisec/(1000 * 60)) % 60)
    microsec = math.modf(millisec)[0] * 1000
    nanosec = math.modf(microsec)[0] * 1000
    print(f"Testing time: {int(minutes)} minutes {int(seconds)} seconds {int(millisec)} milliseconds {int(microsec)} microseconds {int(nanosec)} nanoseconds")


if __name__ == "__main__":
    print("Started 1R algorithm.")
    begin = time.time()
    file = sys.argv[1]
    train_data, test_data, features, outcome = read_data(file)
    rule, rule_score = best_predictor(train_data, features, outcome)
    print("Results: \n")
    print_results(begin, rule, rule_score, train_data, test_data, outcome)

