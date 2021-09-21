
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import gzip
import random
from keras.datasets import mnist

from tensorflow import keras
from tensorflow.keras import layers

def first_weird_work():
    file = './data/train-images-idx3-ubyte.gz'
    f = gzip.open(file,'r')

    image_size = 28
    num_images = 5

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    image = np.asarray(data[2]).squeeze()
    plt.imshow(image)
    plt.show()


def another_read():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    test_X = test_X[:50]
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  ' + str(test_X.shape))
    print('Y_test:  ' + str(test_y.shape))

    test_halves = []
    train_halves = []
    test_match_dict = {}
    train_match_dict = {}

    cut_ammount = 1
    train_y = train_y[:int(len(train_y)/cut_ammount)]


    for i in range(int(len(train_X)/cut_ammount)):
        image = np.array(train_X[i])
        first_half, second_half = np.hsplit(image,[14])

        train_halves.append(first_half)
        train_halves.append(second_half)

        train_match_dict[len(train_halves) - 1] = len(train_halves) - 2
        train_match_dict[len(train_halves) - 2] = len(train_halves) - 1
        # plt.imshow(first_half)
        # plt.show()

    for i in range(len(test_X)):
        image = np.array(train_X[i])
        first_half, second_half = np.hsplit(image, [14])

        test_halves.append(first_half)
        test_halves.append(second_half)

        test_match_dict[len(test_halves) - 1] = len(test_halves) - 2
        test_match_dict[len(test_halves) - 2] = len(test_halves) - 1

    return test_halves, train_halves, test_match_dict, train_match_dict, train_y


def solve_by_random(train_halves):
    results_dict = {}
    idx_list = list(range(len(train_halves)))


    while len(idx_list) > 0:
        idx1 = idx_list[random.randrange(0, len(idx_list))]
        idx_list.remove(idx1)
        idx2 = idx_list[random.randrange(0, len(idx_list))]
        idx_list.remove(idx2)
        results_dict[idx1] = [idx2]
        results_dict[idx2] = [idx1]


    return results_dict


def get_difference(img1, img2):
    total_diff = 0
    for x_index in range(len(img1)):
        val1 = int(img1[x_index][-1])
        val2 = int(img2[x_index][0])
        diff = abs(val1 - val2)
        total_diff += diff

    return total_diff


def solve_by_difference(train_halves):


    results_dict = {}
    results_dict_extended = {}

    for idx in range(len(train_halves)):

        min_diff = 1000000
        min_idy = 0

        all_diffs = []
        for idy in range(len(train_halves)):
            if idx != idy:
                diff = get_difference(train_halves[idx], train_halves[idy])
                if diff < min_diff and diff >= 1:
                    min_diff = diff
                    min_idy = idy

                all_diffs.append([diff, idy])

        all_diffs.sort(key = lambda x:x[0], reverse = False)
        all_diffs = all_diffs[:30]

        results_dict[idx] = [min_idy]
        results_dict[min_idy] = [idx]

        for el in all_diffs:
            if el[1] not in results_dict_extended:
                results_dict_extended[el[1]] = [idx]
            else:
                results_dict_extended[el[1]].append(idx)

            if idx not in results_dict_extended:
                results_dict_extended[idx] = [el[1]]
            else:
                results_dict_extended[idx].append(el[1])


    return results_dict, results_dict_extended


def compare_results(orig_dict, result_dict):
    count = len(orig_dict.keys())
    correct_guesses = 0.0
    false_guesses = 0.0
    for key in orig_dict.keys():
        if key not in result_dict.keys():
            false_guesses += 1
        else:
            if orig_dict[key] in result_dict[key]:
                correct_guesses += 1
            else:
                false_guesses += 1
    print("No of data points: ", count)
    print(" We have an accuracy of: ", correct_guesses/count)


def train_model(train_x, train_y):
    x_train = np.array(train_x)
    y_train = train_y
    num_classes = 10

    input_shape = (28, 14, 1)

    print('X_train: ' + str(x_train.shape))
    print('Y_train: ' + str(y_train.shape))

    x_train = x_train.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    y_train = keras.utils.to_categorical(y_train, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 15


    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    return model


def is_left(image):
    if np.sum(image[:, 0]) > 1:
        return False
    return True



def solve_by_model(train_halves, train_y, test_halves, results_dict_extended):
    train_halves1 = [train_halves[i * 2] for i in range(int(len(train_halves)/2))]
    train_halves2 = [train_halves[i * 2 + 1] for i in range(int(len(train_halves)/2))]

    model_left = train_model(train_halves1, train_y)
    model_right = train_model(train_halves2, train_y)

    for key in list(results_dict_extended.keys()):
        # print("key: ", key)
        if is_left(test_halves[key]):
            label_orig = model_left.predict(test_halves[key].reshape((1, 28, 14, 1)))[0]
            # print("orig:", label_orig)

            for current_sec in results_dict_extended[key]:
                label_sec = model_right.predict(test_halves[current_sec].reshape((1, 28, 14, 1)))[0]
                # print("sec:", label_sec)
                if not list(label_sec) == list(label_orig):
                    results_dict_extended[key].remove(current_sec)
                    results_dict_extended[current_sec].remove(key)


        else:
            label_orig = model_right.predict(test_halves[key].reshape((1, 28, 14, 1)))[0]
            # print("orig:", label_orig)
            for current_sec in results_dict_extended[key]:
                label_sec = model_left.predict(test_halves[current_sec].reshape((1, 28, 14, 1)))[0]
                # print("sec:", label_sec)
                if not list(label_sec) == list(label_orig):
                    results_dict_extended[key].remove(current_sec)
                    results_dict_extended[current_sec].remove(key)

    result_dict = {}

    topx = 5
    for key in list(results_dict_extended.keys()):
        if results_dict_extended[key] == []:
            del results_dict_extended[key]
            print("DELETED A DATA POINT")
        elif len(results_dict_extended[key]) >= topx:
            results_dict_extended[key] = results_dict_extended[key][:topx]

            result_dict[key] = [results_dict_extended[key][0]]
        else:
            print("TOO FEW CHOICES IN A DATA POINT")
    print(result_dict)
    print(results_dict_extended)

    return result_dict, results_dict_extended





def main():
    print("_" * 10)
    print("Reading data points..")
    test_halves, train_halves, test_match_dict, train_match_dict,train_y = another_read()


    # random method usually gets close to 0 correct results
    print("_" * 10)
    print("Solving by random method..")
    result_dict = solve_by_random(test_halves)


    # difference method usually gets close to 0 correct results
    print("_" * 10)
    print("Solving by difference method..")
    result_dict, results_dict_extended = solve_by_difference(test_halves)


    # this method get 10-25% right on first choice , 41-75% right on top three
    print("_" * 10)
    print("Calculating extended results..")
    compare_results(test_match_dict, results_dict_extended)
    print("Calculating results..")
    compare_results(test_match_dict, result_dict)



    # train a simple model
    result_dict, results_dict_extended = solve_by_model(train_halves, train_y, test_halves, results_dict_extended)


    # this improved method get 25% right on first choice , 58-70% right on top three/ five
    print("_" * 10)
    print("Calculating extended results..")
    compare_results(test_match_dict, results_dict_extended)
    print("Calculating results..")
    compare_results(test_match_dict, result_dict)


    debug(results_dict_extended, train_halves)


def debug(results,train_halves):
    for key in list(results.keys())[:7]:
        print("original:")

        plt.imshow(train_halves[key])
        plt.show()

        print("first choice:")
        plt.imshow(train_halves[results[key][0]])
        plt.show()

        print("second choice:")
        plt.imshow(train_halves[results[key][1]])
        plt.show()

        print("third choice:")
        plt.imshow(train_halves[results[key][2]])
        plt.show()


main()