###################################
# CS B551 Fall 2017, Assignment #3
#
# Authors: Aishwarya Dhage(adhage), Ninaad Joshi(ninjoshi), Siddharth Pathak(sidpath)
#
# (Based on skeleton code by D. Crandall)
#
#
####
# For predicting POS tags, HMM is very useful, as the POS tags of words depend on the context and the words which come before
# and after the current word i.e the the tag of word2 depends on what the word1 is and word3 is.
# HMM handles this nicely, as it takes into consideration the transistion probability between words based on the HMM model.
# For predicting the POS tags of words in a sentence, first we have trained our classifier.
# For storing training probabilities, we have used dictionary for faster access. After reading the file data as a list of tuples,
# converted it to dictionary for faster search.
# Different dictionaries store different probabilites like initial_probability, emission probability and transistion probability.
# We calculated initial probability as (number of times a tag appears at the first location divided by the total number of first location tags )
# For Emission probability we calculated it using number of times a word appears in tag divided by total number of words in that tag
# For Transition probability we calculated it using number of outbound transition from a tag to a particular tag
# divided by total number of outbound tags from that tag.
# We stored it as a (noun,verb) as key of dictionary and value as probability.
# After training, each algorithm uses this training data to classify a word's tag.
# Simplified algorithm classifies the tags based on bayes net 1(b), and naive bayes assumption.
# If emission probability is not present then punish it by dividing it by total words present in the corpus.
# Similary, Viterbi uses emission probability, initial probability and transistion probability.
# In Viterbi algorithm, if the transisiton probability is not present then I punished it with factor of 10**-6, and
# if emission probability is not present then punish by dividing it by total words in the corpus.


# ----OUTPUT of BC.test-----


#                           : it's late and  you  said they'd be   here by   dawn ''   .
#  0. Ground truth ( -78.52): prt  adv  conj pron verb prt    verb adv  adp  noun .    .
#    1. Simplified ( -77.41): prt  adj  conj pron verb prt    verb adv  adp  noun .    .
#        2. HMM VE ( -77.41): prt  adj  conj pron verb prt    verb adv  adp  noun .    .
#       3. HMM MAP ( -77.41): prt  adj  conj pron verb prt    verb adv  adp  noun .    .
#
# ==> So far scored 2000 sentences with 29442 words.
#                    Words correct:     Sentences correct:
#    0. Ground truth:      100.00%              100.00%
#      1. Simplified:       93.92%               47.45%
#          2. HMM VE:       95.09%               54.40%
#         3. HMM MAP:       94.88%               53.55%

import random
import math
import copy


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
words_count = {}    # stores the count of the words. Key is a tag and Value is a list. [tag count,dictionary of word and their count]
words_probability = {}  # stores the probability instead of count
initial_probability = {}    # stores initial probability of each tag
outbound_count = {}         # stores the outbound count for each tag
trans = {}                  # stores transition probability for each tag in form of (s1,s2): probability


class Solver:

    def __init__(self):
        self.total_words = 0

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        sum = 0.0
        for i, word in enumerate(sentence):
            if i == 0:
                sum += math.log(initial_probability[label[0]])+math.log(words_probability[label[0]][1].get(word, 10**-8))
            else:
                sum += math.log(words_probability[label[i]][1].get(word, 10 ** -8))+math.log(trans.get((label[i-1], label[i]), 10**-8))
        return sum

    # Do the training!
    #
    def train(self, data):
        # count the words and their tags
        self.total_words = 0
        total_sentences = 0
        for pos in data:
            sentence, tags = pos
            total_sentences += 1
            initial_probability[tags[0]] = initial_probability.get(tags[0], 0) + 1
            for i, word in enumerate(sentence):
                if tags[i] not in words_count:
                    words_count[tags[i]] = [0, {}]
                self.total_words += 1
                words_count[tags[i]][0] += 1
                words_count[tags[i]][1][word] = words_count[tags[i]][1].get(word, 0) + 1
        # calculate P(Wi|Si)
        for tag, count_list in words_count.items():
            words_probability[tag] = [count_list[0]*1.0/self.total_words, {}]
            for word, count in count_list[1].items():
                words_probability[tag][1][word] = count*1.0/count_list[0]
        # calculate P(S1)
        for tag, initial in initial_probability.items():
            initial_probability[tag] = initial_probability[tag]*1.0/total_sentences
        # transition probability
        for pos in data:
            sentence, tag = pos
            for i in range(0, len(tag)-1):
                outbound_count[tag[i]] = outbound_count.get(tag[i], 0) + 1
                trans[(tag[i], tag[i+1])] = trans.get((tag[i], tag[i+1]), 0) + 1
        for k, v in trans.items():
            trans[k] = trans[k]*1.0/outbound_count[k[0]]
            # Functions for each algorithm.
            #
        print len(words_probability)

    def simplified(self, sentence):
        result = []
        temp = {}
        for word in sentence:
            for pos in words_probability:
                temp[pos] = (words_probability[pos][1].get(word, (1.0/self.total_words))*words_probability[pos][0])
            result.append(max(temp, key=lambda x: temp[x]))
            temp.clear()
        return result
        # return [ "p" ] * len(sentence)

    def hmm_ve(self, sentence):
        alpha_prev = {}
        alpha = []
        # forward algorithm
        for i, word in enumerate(sentence):
            alpha_curr = {}
            for pos in words_probability:
                if i == 0:
                    alpha_curr_sum = (initial_probability[pos])
                else:
                    alpha_curr_sum = sum((trans.get((pos1, pos), 10 ** -8))*(alpha_prev[pos1]) for pos1 in words_probability)
                alpha_curr[pos] = alpha_curr_sum*(words_probability[pos][1].get(word, 10 ** -8))
            alpha.append(alpha_curr)
            alpha_prev = alpha_curr
        # backward algorithm
        beta = []
        b_prev = {}
        new_list = sentence[::-1]
        for i, word in enumerate(sentence[::-1]):
            beta_curr = {}
            for pos in words_probability:
                if i == 0:
                    beta_curr[pos] = 1
                else:
                    new_word = new_list[i-1]
                    beta_curr[pos] = sum((trans.get((pos, pos1), 10 ** -8))*(words_probability[pos1][1].get(new_word, 10 ** -8))*((b_prev[pos1]))for pos1 in words_probability)
            beta.append(beta_curr)
            b_prev = beta_curr
        # merge-forward, backward algorithm
        posterior = []
        bfw_dict = {}
        sentence_length = len(sentence)
        j = sentence_length-1
        for i in range(sentence_length):
            for pos in words_probability:
                bfw_dict[pos] = (beta[j][pos] * alpha[i][pos])
            posterior.append(max(bfw_dict, key=bfw_dict.get))
            bfw_dict = {}
            j -= 1
        return posterior

    def hmm_viterbi(self, sentence):
        viterbi = {}
        temp = []
        viterbi_temp = {}
        for i, word in enumerate(sentence):
            if i == 0:
                for j in words_probability:
                    viterbi[j] = [-math.log(initial_probability[j]) + (-math.log(words_probability[j][1].get(word, (1.0/self.total_words)))), [j]]
            else:
                for j in words_probability:
                    for k in viterbi:
                        temp.append([(viterbi[k][0] + (-math.log(trans.get((k, j), 10**-6)))), viterbi[k][1] + [j]])
                    x = min(temp, key=lambda y: y[0])
                    viterbi_temp[j] = [x[0] + (-math.log(words_probability[j][1].get(word, (1.0/self.total_words)))), x[1]]
                    del temp[:]
                viterbi.clear()
                viterbi = copy.deepcopy(viterbi_temp)
                viterbi_temp.clear()
        return min(viterbi.values(), key=lambda v: v[0])[1]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algorithm!"
