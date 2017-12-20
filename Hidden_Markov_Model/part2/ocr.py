#!/usr/bin/python
# coding=utf-8

#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors:Aishwarya Dhage(adhage), Ninaad Joshi(ninjoshi), Siddharth Pathak(sidpath)
# (based on skeleton code by D. Crandall, Oct 2017)
#

from PIL import Image, ImageDraw, ImageFont
import sys

import re

import math

import copy

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH,
                       CHARACTER_WIDTH):
        result += [[1 if px[x, y] < 1 else 0 for x in
                    range(x_beg, x_beg + CHARACTER_WIDTH) for y in
                    range(0, CHARACTER_HEIGHT)]]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS = \
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()," \
                    ".-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in
            range(0, len(TRAIN_LETTERS))}


def read_data(fname):
    file = open(fname, 'r')
    total_character_count = 0
    number_of_lines = 0
    for line in file:
        data = [w for w in line.split()][0::2]
        data = " ".join(data)
        data = re.sub("[^A-Za-z0-9.,\-\(\)!? \"']", "", data)
        data = list(data)  # make list from string
        len_data = len(data)
        total_character_count += len_data
        number_of_lines += 1
        if len_data > 0:
            for char in data:
                all_character_count_table[char] = all_character_count_table.get(char, 0) + 1
            for index in range(0, len_data - 1):
                succeeding_character_count_table[data[index]] = succeeding_character_count_table.get(data[index], 0) + 1
                transition_probability_table[data[index], data[index + 1]] = transition_probability_table.get((data[index], data[index + 1]), 0) + 1
            initial_count_table[data[0]] = initial_count_table.get(data[0], 0) + 1
    
    # calculate all character probability
    for char in all_character_count_table:
        all_character_probability_table[char] = (all_character_count_table[char] * 1.0)/total_character_count
    
    # calculate initial probability
    for initial in initial_count_table:
        initial_probability_table[initial] = (initial_count_table[initial] * 1.0)/number_of_lines
    
    # calculate transition probability
    for k, v in transition_probability_table.items():
        transition_probability_table[k] = (transition_probability_table[k] * 1.0)/succeeding_character_count_table[k[0]]


def calculate_emission_probability(sequence):
    emission_probability_table = {}
    train_sum = sum(
        [sum(train_letters[train_letter]) for train_letter in train_letters])
    test_sum = sum([sum(test_letter) for test_letter in test_letters])
    # print train_sum, test_sum
    
    train_avg = train_sum / len(train_letters)
    test_avg = test_sum / len(test_letters)
    sparse_condition = train_avg / 2
    dense_condition = train_avg
    if 0 <= test_avg <= sparse_condition:
        for i, block in enumerate(sequence):
            emission_probability_table[i] = {}
            for train_letter in train_letters:
                for j, pixel in enumerate(block):
                    if pixel == train_letters[train_letter][j]:
                        if pixel:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.99
                        else:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.5
                    else:
                        if not train_letters[train_letter][j]:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.01
                        else:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.3
    elif sparse_condition < test_avg <= dense_condition:
        for i, block in enumerate(sequence):
            emission_probability_table[i] = {}
            for train_letter in train_letters:
                for j, pixel in enumerate(block):
                    if pixel == train_letters[train_letter][j]:
                        if pixel:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.99
                        else:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.5
                    else:
                        if not train_letters[train_letter][j]:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.01
                        else:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.2
    else:
        for i, block in enumerate(sequence):
            emission_probability_table[i] = {}
            for train_letter in train_letters:
                for j, pixel in enumerate(block):
                    if pixel == train_letters[train_letter][j]:
                        if pixel:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.8
                        else:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.7
                    else:
                        if not train_letters[train_letter][j]:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.2
                        else:
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.3
    return emission_probability_table


def hmm_viterbi(test_letters):
    viterbi = {}
    temp = []
    viterbi_temp = {}
    for i, word in enumerate(test_letters):
        if i == 0:
            for j in train_letters:
                viterbi[j] = [-math.log(initial_probability_table.get(j, 10 ** -6)) + (-math.log(emission_probability_table[i].get(j, (10 ** -6)))), [j]]
        else:
            for j in train_letters:
                for k in viterbi:
                    temp.append([(viterbi[k][0] + (-math.log(transition_probability_table.get((k, j), 10 ** -6)))), viterbi[k][1] + [j]])
                x = min(temp, key=lambda y: y[0])
                viterbi_temp[j] = [x[0] + (-math.log(emission_probability_table[i].get(j, 10 ** -6))), x[1]]
                del temp[:]
            viterbi.clear()
            viterbi = copy.deepcopy(viterbi_temp)
            viterbi_temp.clear()
    # print viterbi
    return min(viterbi.values(), key=lambda y: y[0])[1]


def hmm_ve(sequence):
    alpha_prev = {}
    alpha = []
    # forward algorithm
    for i, word in enumerate(test_letters):
        alpha_curr = {}
        
        for pos in train_letters:
            if i == 0:
                alpha_curr_sum = (initial_probability_table.get(pos, 10 ** -3))
            else:
                alpha_curr_sum = sum(
                    (transition_probability_table.get((pos1, pos), 0.0001)) * (alpha_prev[pos1]) for pos1 in train_letters)
            alpha_curr[pos] = (alpha_curr_sum * (emission_probability_table[i].get(pos, 10 ** -3)))
        temp_max = max(alpha_curr.values())
        new_alpha = {key: val / temp_max for key, val in alpha_curr.iteritems()}
        alpha.append(new_alpha)
        alpha_prev = new_alpha
    # backward algorithm
    beta = []
    b_prev = {}
    prev_letter = len(test_letters) - 1
    for i, word in enumerate(test_letters[::-1]):
        beta_curr = {}
        for pos in train_letters:
            if i == 0:
                beta_curr[pos] = 1
            else:
                beta_curr[pos] = sum((transition_probability_table.get(
                    (pos, pos1), 10 ** -3)) * (emission_probability_table[prev_letter].get(pos1, 10 ** -3)) * (b_prev[pos1]) for pos1 in train_letters)
        temp_max = max(beta_curr.values())
        new_beta = {key: val / temp_max for key, val in beta_curr.iteritems()}
        beta.append(new_beta)
        b_prev = new_beta
        prev_letter -= 1
    # merge-forward, backward algorithm
    posterior = []
    bfw_dict = {}
    sentence_length = len(test_letters)
    j = sentence_length - 1
    for i in range(sentence_length):
        for pos in train_letters:
            bfw_dict[pos] = (beta[j][pos] * alpha[i][pos])
        posterior.append(max(bfw_dict, key=bfw_dict.get))
        bfw_dict = {}
        j -= 1
    return posterior


(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
initial_probability_table = {}
initial_count_table = {}
all_character_count_table = {}
all_character_probability_table = {}
succeeding_character_count_table = {}
transition_probability_table = {}

try:
    read_data(train_txt_fname)
    emission_probability_table = calculate_emission_probability(test_letters)
    result = []
    for k, v in emission_probability_table.items():
        result.append(max(v, key=v.get))
    print "".join(result)
    print "".join(hmm_ve(test_letters))
    # emission_probability_table = calculate_emission_probability(test_letters)
    print "".join(hmm_viterbi(test_letters))

except IOError:
    print "File not found :("
    exit(0)
