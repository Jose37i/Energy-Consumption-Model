import math
import random
import numpy as np
import csv
import matplotlib.pyplot as plt


def normalize_it(passed_data):
    hour_arr = passed_data[0]
    kw_arr = passed_data[1]
    bias_arr = passed_data[2]
    min_hour = min(hour_arr)
    max_hour = max(hour_arr)
    min_kw = min(kw_arr)
    max_kw = max(kw_arr)

    for index in range(len(hour_arr)):
        hour_arr[index] = (hour_arr[index] - min_hour) / (max_hour - min_hour)
        kw_arr[index] = (kw_arr[index] - min_kw) / (max_kw - min_kw)

    passed_data = [hour_arr, kw_arr, bias_arr]
    return passed_data


def read_files(normalize):
    files = ['train_data_1.txt', 'train_data_2.txt', 'train_data_3.txt', 'test_data_4.txt']
    resulting_training_data = [[], [], []]
    resulting_testing_data = [[], [], []]
    for entry in files:
        with open(entry, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            if entry != 'test_data_4.txt':
                for line in csv_reader:
                    resulting_training_data[0].append(float(line[0]))
                    resulting_training_data[1].append(float(line[1]))
                    resulting_training_data[2].append(1.0)
            else:
                for line in csv_reader:
                    resulting_testing_data[0].append(float(line[0]))
                    resulting_testing_data[1].append(float(line[1]))
                    resulting_testing_data[2].append(1.0)
    if not normalize:
        return resulting_training_data, resulting_testing_data
    else:
        resulting_training_data = normalize_it(resulting_training_data)
        resulting_testing_data = normalize_it(resulting_testing_data)
        return resulting_training_data, resulting_testing_data


train_data, test_data = read_files(False)
test_vals = np.array(test_data)
print(test_vals)
