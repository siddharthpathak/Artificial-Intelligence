# B-551 Fall 2017
# Assignment 2, Question 2
# Tweet Classification using multinomial Naive Bayes Classifier
# The training data contained a lot of new line and carriage return characters.
# To handle that, I used strip and split functions provided by Python.
# Strip functions removes all the trailing and leading white spaces.
# Split function splits the words by the given argument into a list.
# If the second argument is None, then it splits based by whitespace characters(tab, newline, carraige return, etc) (https://docs.python.org/2/library/string.html#string.split)
# So, it removed all the whitespace characters while converting it into a list.
# Few words in the training data set contained the words with non ASCII characters or special characters.
# To remove the Non ASCII characters I used regex. If the words contain non letter or non numbers then remove them, along with removing their capitalization.
# Also, few of the words are common in all the cities. Those words don't provide any data significant data about the given tweet.
# We referred the following list for stop words in English : "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
# We ignored those words by making a dictionary "stop words", and it is checked before new probability is calculated.
# Few of the tweets also contain single letters. Like one of the tweets has "F R I D A Y" in it. While parsing it reads it as single character for each word.
# I added single letters to the stop words dictionary too.
# Furthermore, few of the tweets contain special character "&" as "&amp". Regex removes the "&" from the token but keeps the "amp" as a token. So, we added "amp" as a stop word.
# Tweets containing tokens "jobs", "job", and "hiring" are also present in all cities, so we added them to the stop word list too.
# For calculating posterior probability, if the word is not present in the current city's training data, then we punished it using the minimum probability possible (psuedo count)
# The minimum probability possible for a word is (1/total number of words). So, the possibility that the tweet belongs to that city, decreases as that city didn't contain that word in the training data.
# If the test data just contains the city name and doesn't contain any tokens then we classified it as "No_Data"

import re
import sys

total_tweets = 0
city_words = {}
city_words_pro = {}
total_words = 0
posterior = {}
original_cities = []
result = []
stop_words = {'all': 1, 'just': 1, 'being': 1, 'over': 1, 'both': 1, 'through': 1, 'yourselves': 1, 'its': 1,
              'before': 1, 'herself': 1, 'had': 1, 'should': 1, 'to': 1, 'only': 1, 'under': 1, 'ours': 1, 'has': 1, 'do': 1, 'them': 1, 'his': 1, 'very': 1,
              'they': 1, 'not': 1, 'during': 1, 'now': 1, 'him': 1, 'nor': 1, 'did': 1, 'this': 1, 'she': 1, 'each': 1, 'further': 1, 'where': 1, 'few': 1, 'because': 1,
              'doing': 1, 'some': 1, 'are': 1, 'our': 1, 'ourselves': 1, 'out': 1, 'what': 1, 'for': 1, 'while': 1, 'does': 1, 'above': 1, 'between': 1, 't': 1, 'be': 1,
              'we': 1, 'who': 1, 'were': 1, 'here': 1, 'hers': 1, 'by': 1, 'on': 1, 'about': 1, 'of': 1, 'against': 1, 's': 1, 'or': 1, 'own': 1, 'into': 1, 'yourself': 1,
              'down': 1, 'your': 1, 'from': 1, 'her': 1, 'their': 1, 'there': 1, 'been': 1, 'whom': 1, 'too': 1, 'themselves': 1, 'was': 1, 'until': 1, 'more': 1, 'himself': 1,
              'that': 1, 'but': 1, 'don': 1, 'with': 1, 'than': 1, 'those': 1, 'he': 1, 'me': 1, 'myself': 1, 'these': 1, 'up': 1, 'will': 1, 'below': 1, 'can': 1, 'theirs': 1,
              'my': 1, 'and': 1, 'then': 1, 'is': 1, 'am': 1, 'it': 1, 'an': 1, 'as': 1, 'itself': 1, 'at': 1, 'have': 1, 'in': 1, 'any': 1, 'if': 1, 'again': 1, 'no': 1, 'when': 1,
              'same': 1, 'how': 1, 'other': 1, 'which': 1, 'you': 1, 'after': 1, 'most': 1, 'such': 1, 'why': 1, 'a': 1, 'off': 1, 'i': 1, 'yours': 1, 'so': 1, 'the': 1, 'having': 1,
              'once': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1, 'h': 1, 'i': 1, 'j': 1, 'k': 1, 'l': 1, 'm': 1, 'n': 1,'o': 1,'p': 1,'q': 1,'r': 1,'s': 1,'t': 1,'u': 1,
              'v': 1, 'w': 1, 'x': 1, 'y': 1, 'z': 1, 'job': 1, 'im': 1, 'amp': 1, 'jobs': 1, 'hiring': 1}


def train_classifier(training_file):
    global total_words
    total_words = 0
    global total_tweets
    total_tweets = 0
    for line in open(training_file):
        temp_line = line.strip().split()                # split and strip the line so as to
        if temp_line[0] not in city_words:
            city_words[temp_line[0]] = [0, 0, {}]
        city_words[temp_line[0]][0] += 1                # total number of tweets per city at 0th index
        for word in temp_line[1:]:
            word = re.sub('[^A-Za-z0-9]', '', word.lower())     # remove characters and special symbols
            if word != '' and word not in stop_words:
                total_words += 1
                city_words[temp_line[0]][1] += 1        # total number of words in the city
                city_words[temp_line[0]][2][word] = city_words[temp_line[0]][2].get(word, 0) + 1
        total_tweets += 1
    for k, v in city_words.items():
        city_words_pro[k] = [v[0], {}]            # store number of tweets per city, and word probability for each word
        for words, words_count in v[2].items():                   # iterate over each word count
            city_words_pro[k][1][words] = words_count*1.0/v[1]    # store each word count divide by total number of words for that city


def tweet_classifier(testing_file, output_file):
    output_file_handler = open(output_file, "a")
    for line in open(testing_file):
        temp_line = line.strip().split()
        original_cities.append(temp_line[0])
        for word in temp_line[1:]:
            word = re.sub('[^A-Za-z0-9]', '', word.lower())    # remove characters and special symbols
            if word != '' and word not in stop_words:
                for cities in city_words_pro.keys():
                    if city_words_pro[cities][1].get(word, None) is None:
                        punish = 1.0/total_words               # punish by psuedo count 1.0/total_words
                    else:
                        punish = city_words_pro[cities][1][word]
                    posterior[cities] = posterior.get(cities, city_words_pro[cities][0]*1.0/total_tweets) * punish
        if not posterior:
            result.append("No_Data")            # if the tweet is empty then we can't classify as there are no tokens
            output_file_handler.write("No_Data" + " " + line)
        else:
            result.append(max(posterior, key=lambda x: posterior[x]))
            output_file_handler.write((max(posterior, key=lambda x: posterior[x]))+" " + line)
        posterior.clear()
    difference = 0
    for i in range(0, len(result)):
        if original_cities[i] == result[i]:
            difference += 1
    for k, v in city_words.items():
        print k, ": ", ",".join(sorted(v[2], key=lambda x: v[2][x], reverse=True)[:5])


training_file, testing_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

train_classifier(training_file)
tweet_classifier(testing_file, output_file)