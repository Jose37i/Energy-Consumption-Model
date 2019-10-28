import math
import random
import numpy as np
import csv
import matplotlib.pyplot as plt


def normalize_it(passed_data):
    hour_arr = passed_data[1]
    kw_arr = passed_data[2]
    bias_arr = passed_data[0]
    min_hour = min(hour_arr)
    max_hour = max(hour_arr)
    min_kw = min(kw_arr)
    max_kw = max(kw_arr)

    for index in range(len(hour_arr)):
        hour_arr[index] = (hour_arr[index] - min_hour) / (max_hour - min_hour)
        kw_arr[index] = (kw_arr[index] - min_kw) / (max_kw - min_kw)

    passed_data = [bias_arr, hour_arr, kw_arr]
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
                    resulting_training_data[0].append(1.0)
                    resulting_training_data[1].append(float(line[0]))
                    resulting_training_data[2].append(float(line[1]))
            else:
                for line in csv_reader:
                    resulting_testing_data[0].append(1.0)
                    resulting_testing_data[1].append(float(line[0]))
                    resulting_testing_data[2].append(float(line[1]))
    if not normalize:
        return resulting_training_data, resulting_testing_data
    else:
        resulting_training_data = normalize_it(resulting_training_data)
        resulting_testing_data = normalize_it(resulting_testing_data)
        return resulting_training_data, resulting_testing_data


def generate_weights(amount_needed):
    generated_weights = []
    for num in range(amount_needed):
        generated_weights.append(random.uniform(-.5, .5))

    return generated_weights


def square_values(passed_values, to_the_power_of):
    bias_values = passed_values[0]
    hour_values = passed_values[1]
    resulting_array = [bias_values, hour_values]
    if to_the_power_of == 3:
        hour_values_squared = passed_values[2]
        kw_values = passed_values[3]
        resulting_array.append(hour_values_squared)
    else:
        kw_values = passed_values[2]
    temp_new_hours = []
    for d in range(len(hour_values)):
        temp_new_hours.append(pow(hour_values[d], to_the_power_of))
    resulting_array.append(temp_new_hours)
    resulting_array.append(kw_values)

    return resulting_array


def activation_function(data_point, passed_weight):
    net_val = 0
    for l in range(len(passed_weight)):
        net_val += data_point[l] * passed_weight[l]
    return net_val


def train(passed_data, p_weights, alpha, constant):
    # All kw values
    kw_values = passed_data[len(p_weights)]
    # the length of the values  which is 48 , 3 days combined
    values_length = passed_data[0]
    length = len(values_length)
    # The length of the weights list
    weights_length = len(p_weights)
    # doing _ iterations
    for iteration in range(300):
        # for the length of the values which is 48
        for t in range(length):
            # declare inputs list
            inputs = []
            # for the length of the weights
            for m in range(weights_length):
                # append the input value corresponding to weight
                inputs.append(passed_data[m][t])
            # out = activvation function
            # inputs ex. [1, 5, 3.71]
            # weight sex. [-0.5, 0.5, -0.5]
            out = activation_function(inputs, weights)
            # error = the predcited kw value - the out
            error = float(kw_values[t]) - out
            print(error)
            # adjust the weights
            for v in range(weights_length):
                p_weights[v] += alpha * error * passed_data[v][t]
    return p_weights


train_data, test_data = read_files(False)
weights = generate_weights(2)
# train_data = square_values(train_data, 2)
weights = train(train_data, weights, 0.3, 1)
print(weights)



# train(train_data, weights, 0.1, 1)
# print(generate_weights(2))
# test_vals = np.array(test_data)
# print(test_vals)
