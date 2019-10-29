import math
import random
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
        # hour_arr[index] = (hour_arr[index] - min_hour) / (max_hour - min_hour)
        hour_arr[index] /= max_hour
        kw_arr[index] /= max_kw
        # kw_arr[index] = (kw_arr[index] - min_kw) / (max_kw - min_kw)

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


def read_file(normalize, file_name):
    resulting_training_data = [[], [], []]
    resulting_testing_data = [[], [], []]
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            resulting_training_data[0].append(1.0)
            resulting_training_data[1].append(float(line[0]))
            resulting_training_data[2].append(float(line[1]))
    if not normalize:
        return resulting_training_data, resulting_testing_data
    else:
        return normalize_it(resulting_training_data)


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


def train(passed_data, p_weights, alpha):
    # All kw values
    kw_values = passed_data[len(p_weights)]
    # the length of the values  which is 48 , 3 days combined
    values_length = passed_data[0]
    length = len(values_length)
    # The length of the weights list
    weights_length = len(p_weights)
    # doing _ iterations
    for iteration in range(50000):
        # for the length of the values which is 48
        total_error_calc = 0
        for t in range(length):
            # declare inputs list
            inputs = []
            # for the length of the weights
            for m in range(weights_length):
                # append the input value corresponding to weight
                inputs.append(passed_data[m][t])
            # out = activation function
            # inputs ex. [1, 5, 3.71]
            # weight ex. [-0.5, 0.5, -0.5]
            out = activation_function(inputs, p_weights)
            # error = the predicted kw value - the out
            error = float(kw_values[t]) - out
            total_error_calc += math.pow(error, 2)
            # adjust the weights
            for v in range(weights_length):
                p_weights[v] += alpha * error * passed_data[v][t]
    return p_weights


def tester(tester_data, weight_set):
    total_error = 0
    kw_actual = tester_data[-1]
    size = len(tester_data[0])
    for x in range(0, size):
        prediction = 0
        for r in range(0, len(weight_set)):
            prediction += tester_data[r][x] * weight_set[r]
        total_error += (kw_actual[x] - prediction) ** 2
    return total_error


def make_plot(passed_points, p_weight_set, p_title, p_filename):
    hours = passed_points[1].copy()
    kw = passed_points[-1].copy()
    temp_hours = passed_points[1].copy()
    predicted_points = []
    plot = plt.figure()
    plot.suptitle(p_title)

    for j in range(0, len(hours)):
        temp = []
        for f in range(0, len(p_weight_set)):
            temp.append(passed_points[f][j])
        predicted_points.append(activation_function(temp, p_weight_set))
        plt.plot(hours[j], predicted_points[j])

    hours_to_predicted_dictionary = {}
    for j in range(len(hours)):
        hours_to_predicted_dictionary[hours[j]] = predicted_points[j]
    hours.sort()
    for t in range(len(hours)):
        val = hours[t]
        predicted_points[t] = hours_to_predicted_dictionary[val]

    plt.xlabel('Hours', fontsize=17)
    plt.ylabel('Kilowatts', fontsize=17)
    plt.scatter(temp_hours, kw)
    plt.plot(hours, predicted_points, '#2fef10')
    plot.savefig(p_filename)


def print_graph_info(neuron_number, p_day_number, p_training_file_name, p_testing_file_name, weights_for_graph,
                     p_train_data, p_test_data):
    if p_day_number != 0:
        print('Neuron ' + str(neuron_number) + ' for day ' + str(p_day_number) + ':')
    else:
        print('Neuron ' + str(neuron_number) + ':')
    print('\tTraining Graph File Name is: ' + p_training_file_name)
    print('\tTesting Graph File Name is: ' + p_testing_file_name)
    print('\tWeights are [Bias, Hour, Hour^2, Hour^3]  ---> ' + str(weights_for_graph))
    print('\tTraining Error:' + str(tester(p_train_data, weights_for_graph)))
    print('\tTesting Error:' + str(tester(p_test_data, weights_for_graph)) + '\n\n')


train_data_day_1 = read_file(True, 'train_data_1.txt')
train_data_day_2 = read_file(True, 'train_data_2.txt')
train_data_day_3 = read_file(True, 'train_data_3.txt')
test_data = read_file(True, 'test_data_4.txt')
neuron1_weights = generate_weights(2)
neuron1_weights_day_1 = train(train_data_day_1.copy(), neuron1_weights.copy(), 0.01)
neuron1_weights_day_2 = train(train_data_day_2.copy(), neuron1_weights.copy(), 0.01)
neuron1_weights_day_3 = train(train_data_day_3.copy(), neuron1_weights.copy(), 0.01)
training_title = 'Neuron 1: Training'
training_file_name = 'Neuron1.Train.'
testing_tile = 'Neuron 1: Testing'
testing_file_name = 'Neuron1.Test.'
make_plot(train_data_day_1, neuron1_weights_day_1, training_title + ' for day 1', training_file_name + 'Day1.png')
make_plot(test_data, neuron1_weights_day_1, testing_tile + ' for day 1', testing_file_name + 'Day1.png')
print_graph_info(1, 1, training_file_name + 'Day1.png', testing_file_name + 'Day1.png', neuron1_weights_day_1,
                 train_data_day_1, test_data)

make_plot(train_data_day_2, neuron1_weights_day_2, training_title + ' for day 2', training_file_name + 'Day2.png')
make_plot(test_data, neuron1_weights_day_2, testing_tile + ' for day 2', testing_file_name + 'Day2.png')
print_graph_info(1, 2, training_file_name + 'Day2.png', testing_file_name + 'Day2.png', neuron1_weights_day_2,
                 train_data_day_2, test_data)

make_plot(train_data_day_3, neuron1_weights_day_3, training_title + ' for day 3', training_file_name + 'Day3.png')
make_plot(test_data, neuron1_weights_day_3, testing_tile + ' for day 3', testing_file_name + 'Day3.png')
print_graph_info(1, 3, training_file_name + 'Day3.png', testing_file_name + 'Day3.png', neuron1_weights_day_3,
                 train_data_day_3, test_data)

neuron2_weights = generate_weights(3)
neuron2_train_data__day_1 = square_values(train_data_day_1.copy(), 2)
neuron2_train_data__day_2 = square_values(train_data_day_2.copy(), 2)
neuron2_train_data__day_3 = square_values(train_data_day_3.copy(), 2)
test_data_neuron_2 = square_values(test_data.copy(), 2)
neuron2_weights_day_1 = train(neuron2_train_data__day_1, neuron2_weights.copy(), 0.2)
neuron2_weights_day_2 = train(neuron2_train_data__day_2, neuron2_weights.copy(), 0.2)
neuron2_weights_day_3 = train(neuron2_train_data__day_3, neuron2_weights.copy(), 0.2)
training_title = 'Neuron 2: Training'
training_file_name = 'Neuron2.Train.'
testing_tile = 'Neuron 2: Testing'
testing_file_name = 'Neuron2.Test.'


make_plot(neuron2_train_data__day_1, neuron2_weights_day_1, training_title + ' for day 1', training_file_name + 'Day1.png')
make_plot(test_data_neuron_2, neuron2_weights_day_1, testing_tile + ' for day 1', testing_file_name + 'Day1.png')
print_graph_info(2, 1, training_file_name + 'Day1.png', testing_file_name + 'Day1.png', neuron2_weights_day_1,
                 neuron2_train_data__day_1, test_data_neuron_2)

make_plot(neuron2_train_data__day_2, neuron2_weights_day_2, training_title + ' for day 2', training_file_name + 'Day2.png')
make_plot(test_data_neuron_2, neuron2_weights_day_2, testing_tile + ' for day 2', testing_file_name + 'Day2.png')
print_graph_info(2, 2, training_file_name + 'Day2.png', testing_file_name + 'Day2.png', neuron2_weights_day_2,
                 neuron2_train_data__day_2, test_data_neuron_2)

make_plot(neuron2_train_data__day_3, neuron2_weights_day_3, training_title + ' for day 3', training_file_name + 'Day3.png')
make_plot(test_data_neuron_2, neuron2_weights_day_3, testing_tile + ' for day 3', testing_file_name + 'Day3.png')
print_graph_info(2, 3, training_file_name + 'Day3.png', testing_file_name + 'Day3.png', neuron2_weights_day_3,
                 neuron2_train_data__day_3, test_data_neuron_2)


neuron3_weights = generate_weights(4)
neuron3_train_data__day_1 = square_values(neuron2_train_data__day_1.copy(), 3)
neuron3_train_data__day_2 = square_values(neuron2_train_data__day_2.copy(), 3)
neuron3_train_data__day_3 = square_values(neuron2_train_data__day_3.copy(), 3)
test_data_neuron_3 = square_values(test_data_neuron_2.copy(), 3)
neuron3_weights_day_1 = train(neuron3_train_data__day_1, neuron3_weights.copy(), 0.2)
neuron3_weights_day_2 = train(neuron3_train_data__day_2, neuron3_weights.copy(), 0.2)
neuron3_weights_day_3 = train(neuron3_train_data__day_3, neuron3_weights.copy(), 0.2)
training_title = 'Neuron 3: Training'
training_file_name = 'Neuron3.Train.'
testing_tile = 'Neuron 3: Testing'
testing_file_name = 'Neuron3.Test.'


make_plot(neuron3_train_data__day_1, neuron3_weights_day_1, training_title + ' for day 1', training_file_name + 'Day1.png')
make_plot(test_data_neuron_3, neuron3_weights_day_1, testing_tile + ' for day 1', testing_file_name + 'Day1.png')
print_graph_info(3, 1, training_file_name + 'Day1.png', testing_file_name + 'Day1.png', neuron3_weights_day_1,
                 neuron3_train_data__day_1, test_data_neuron_3)

make_plot(neuron3_train_data__day_2, neuron3_weights_day_2, training_title + ' for day 2', training_file_name + 'Day2.png')
make_plot(test_data_neuron_3, neuron3_weights_day_2, testing_tile + ' for day 2', testing_file_name + 'Day2.png')
print_graph_info(3, 2, training_file_name + 'Day2.png', testing_file_name + 'Day2.png', neuron3_weights_day_2,
                 neuron3_train_data__day_2, test_data_neuron_3)

make_plot(neuron3_train_data__day_3, neuron3_weights_day_3, training_title + ' for day 3', training_file_name + 'Day3.png')
make_plot(test_data_neuron_3, neuron3_weights_day_3, testing_tile + ' for day 3', testing_file_name + 'Day3.png')
print_graph_info(3, 3, training_file_name + 'Day3.png', testing_file_name + 'Day3.png', neuron3_weights_day_3,
                 neuron3_train_data__day_3, test_data_neuron_3)


train_data, test_data = read_files(True)
neuron1_weights = generate_weights(2)
neuron1_weights = train(train_data, neuron1_weights, 0.01)
training_title = 'Neuron 1: Training with All Days'
training_file_name = 'Neuron1.Train.All.Days.png'
testing_tile = 'Neuron 1: Testing with All Days'
testing_file_name = 'Neuron1.Test.All.Days.png'
make_plot(train_data, neuron1_weights, training_title, training_file_name)
make_plot(test_data, neuron1_weights, testing_tile, testing_file_name)
print_graph_info(1, 0, training_file_name, testing_file_name, neuron1_weights, train_data, test_data)


neuron2_weights = generate_weights(3)
train_data_neuron_2 = square_values(train_data, 2)
test_data_neuron_2 = square_values(test_data, 2)
neuron2_weights = train(train_data_neuron_2, neuron2_weights, 0.2)
training_title = 'Neuron 2: Training with All Days'
training_file_name = 'Neuron2.Train.All.Days.png'
testing_tile = 'Neuron 2: Testing with All Days'
testing_file_name = 'Neuron2.Test.All.Days.png'
make_plot(train_data_neuron_2, neuron2_weights, training_title, training_file_name)
make_plot(test_data_neuron_2, neuron2_weights, testing_tile, testing_file_name)
print_graph_info(2, 0, training_file_name, testing_file_name, neuron2_weights, train_data_neuron_2, test_data_neuron_2)


neuron3_weights = generate_weights(4)
train3_data = square_values(train_data_neuron_2, 3)
test_data_neuron_3 = square_values(test_data_neuron_2, 3)
neuron3_weights = train(train3_data, neuron3_weights, 0.2)
training_title = 'Neuron 3: Training with All Days'
training_file_name = 'Neuron3.Train.All.Days.png'
testing_tile = 'Neuron 3: Testing with All Days'
testing_file_name = 'Neuron3.Test.All.Days.png'
make_plot(train3_data, neuron3_weights, training_title, training_file_name)
make_plot(test_data_neuron_3, neuron3_weights, testing_tile, testing_file_name)
print_graph_info(3, 0, training_file_name, testing_file_name, neuron3_weights, train3_data, test_data_neuron_3)