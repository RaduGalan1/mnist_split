import idx2numpy
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import gzip
import random

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

from keras.datasets import mnist

def another_read():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  ' + str(test_X.shape))
    print('Y_test:  ' + str(test_y.shape))

    test_halves = []
    train_halves = []
    test_match_dict = {}
    train_match_dict = {}

    for i in range(int(len(train_X)/1000)):
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

    return test_halves, train_halves, test_match_dict, train_match_dict


def solve_by_random(train_halves):
    results_dict = {}
    idx_list = list(range(len(train_halves)))


    while len(idx_list) > 0:
        idx1 = idx_list[random.randrange(0, len(idx_list))]
        idx_list.remove(idx1)
        idx2 = idx_list[random.randrange(0, len(idx_list))]
        idx_list.remove(idx2)
        results_dict[idx1] = idx2
        results_dict[idx2] = idx1


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

    for idx in range(len(train_halves)):

        min_diff = 1000000
        min_idy = 0
        for idy in range(len(train_halves)):
            if idx != idy:
                diff = get_difference(train_halves[idx], train_halves[idy])
                if diff < min_diff and diff >= 1:
                    min_diff = diff
                    min_idy = idy

        results_dict[idx] = min_idy
        results_dict[min_idy] = idx

    return results_dict


def compare_results(orig_dict, result_dict):
    count = len(orig_dict.keys())
    correct_guesses = 0.0
    false_guesses = 0.0
    for key in orig_dict.keys():
        if key not in result_dict.keys():
            false_guesses += 1
        else:
            if orig_dict[key] == result_dict[key]:
                correct_guesses += 1
            else:
                false_guesses += 1
    print("No of data points: ", count)
    print(" We have an accuracy of: ", correct_guesses/count)


def main():
    print("Reading data points..")
    test_halves, train_halves, test_match_dict, train_match_dict = another_read()


    # random method usually gets close to 0 correct results
    print("Solving by random method..")
    result_dict = solve_by_random(train_halves)

    # difference method usually gets close to 0 correct results
    print("Solving by difference method..")
    result_dict = solve_by_difference(train_halves)



    print("CaLculating results..")
    compare_results(train_match_dict, result_dict)

main()