import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from random import shuffle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from statistics import mean
import numpy as np
from keras import models
from keras.optimizers import Adam
from keras import layers
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC


def dividing_reviews(parameter, dataset):
    reviews = []
    for i in range(len(dataset.Review.values)):
        if dataset.Liked.values[i] == parameter:
            reviews.append(dataset.Review.values[i])
    return reviews


def filtering_Reviews(reviews):
    # removing the punctuations
    text_nopunct = ''
    text_nopunct = "".join([char for char in reviews if char not in string.punctuation])

    # Creating the tokenizer
    tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
    pos_tokens = tokenizer.tokenize(text_nopunct)
    # making everything lowercase and splitting sentences into words

    words = []
    for word in pos_tokens:
        words.append(word.lower())

    stopwords = nltk.corpus.stopwords.words("english")

    # Removing stopwords
    final_words = []
    for word in words:
        if word not in stopwords:
            final_words.append(word)
    return final_words


def lemmatizeWords_plotFreq_distribution(final_words):
    wn = WordNetLemmatizer()
    lem_words = []
    for word in final_words:
        word = wn.lemmatize(word)
        lem_words.append(word)
    # The frequency distribution of the words
    freq_dist = nltk.FreqDist(lem_words)
    # Frequency Distribution Plot
    plt.subplots(figsize=(20, 12))
    freq_dist.plot(30)
    return freq_dist


def wordCloud(lem_words):
    res = ' '.join([i for i in lem_words if not i.isdigit()])
    plt.subplots(figsize=(16, 10))
    wordcloud = WordCloud(
        background_color='black',
        max_words=100,
        width=1400,
        height=1200
    ).generate(res)
    plt.imshow(wordcloud)
    plt.title('Reviews World Cloud(100 words)')
    plt.axis('off')
    plt.show()


def restaurantReviews():
    stopwords = nltk.corpus.stopwords.words('english')

    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

    # dividing reviews into positive nad negative review
    positive_rev = dividing_reviews(1, dataset)

    negative_rev = dividing_reviews(0, dataset)

    all_rev = positive_rev + negative_rev
    shuffle(all_rev)

    final_words_pos = filtering_Reviews(positive_rev)
    final_words_neg = filtering_Reviews(negative_rev)

    # lemmatize words e.g. (cars, car => car) , (are, were, was => is)
    lem_words_pos = lemmatizeWords_plotFreq_distribution(final_words_pos)
    lem_words_neg = lemmatizeWords_plotFreq_distribution(final_words_neg)

    # Delete common words occurring in intersection
    positive_fd = nltk.FreqDist(lem_words_pos)
    negative_fd = nltk.FreqDist(lem_words_neg)
    common_set = set(lem_words_pos).intersection(lem_words_neg)
    for word in common_set:
        del positive_fd[word]
        del negative_fd[word]

    # Get top 100 positive and top 100 negative words in reviews
    top_100_positive = {word for word, count in positive_fd.most_common(100)}
    top_100_negative = {word for word, count in negative_fd.most_common(100)}

    # Option to add Bigram Collocation feature to the existing dataset
    positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
        w for w in positive_fd
        if w.isalpha() and w not in stopwords
    ])
    negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
        w for w in negative_fd
        if w.isalpha() and w not in stopwords
    ])

    # {'mean_compound': 1.0119444444444445, 'mean_positive': 0.11548148148148148, 'wordcount_count_positive': 3, 'wordcount_count_negative': 1}
    def extract_features(text):
        wordcount_pos = 0
        wordcount_neg = 0
        bigram_count_pos = 0
        bigram_count_neg = 0
        compound_scores = list()
        positive_scores = list()
        sia = SentimentIntensityAnalyzer()
        for sentence in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sentence):
                if word.lower() in top_100_positive:
                    wordcount_pos += 1
                if word.lower() in top_100_negative:
                    wordcount_neg += 1
                if word in positive_bigram_finder.word_fd:
                    bigram_count_pos += 1
                if word in negative_bigram_finder.word_fd:
                    bigram_count_neg += 1
            compound_scores.append(sia.polarity_scores(sentence)["compound"])
            positive_scores.append(sia.polarity_scores(sentence)["pos"])

        # Adding 1 to the final compound score to always have positive numbers
        # since some classifiers you'll use later don't work with negative numbers.
        curr_features = [mean(compound_scores) + 1, mean(positive_scores), wordcount_pos, wordcount_neg, bigram_count_pos, bigram_count_neg]
        return curr_features

    corpus = []
    lemmatizer = WordNetLemmatizer()
    y = []
    y_neural = []
    for i in range(0, 1000):  # as the data as 1000 data points
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords) and word.isalpha()]
        review = ' '.join(review)
        if len(review) > 0:
            corpus.append(review)
            y.append(dataset.values[i][1])
            if dataset.values[i][1] == 1:
                y_neural.append([1, 0])
            else:
                y_neural.append([0, 1])

    # Creating the Bag of Words model
    cv = CountVectorizer(max_features=2000)
    # the X and y
    X = cv.fit_transform(corpus).toarray()
    new_features = []
    for row in range(len(X)):
        curr_added_features = extract_features(corpus[row])
        new_features.append(curr_added_features)
    X = np.append(X, new_features, axis=1)

    # y = dataset.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

    # <--------------------------------------------------------- Neural Network ------------------------------------->
    # X_train_neural, X_test_neural, y_train_neural, y_test_neural = train_test_split(X, y_neural, test_size=0.25,
    #                                                                                 random_state=7)
    #
    # model_2B = models.Sequential()
    # model_2B.add(layers.Dense(1772, activation='relu'))
    # model_2B.add(layers.Dense(1000, activation='relu'))
    # model_2B.add(layers.Dense(500, activation='relu'))
    # model_2B.add(layers.Dense(250, activation='relu'))
    # model_2B.add(layers.Dense(100, activation='softmax'))
    # model_2B.add(layers.Dense(10, activation='softmax'))
    # model_2B.add(layers.Dense(2, activation='softmax'))
    # model_2B.compile(optimizer='adam',
    #                  loss='binary_crossentropy',
    #                  metrics=['acc'])
    # neural_X_train = np.asarray(X_train_neural).astype('float32')
    # neural_Y_train = np.asarray(y_train_neural).astype('float32')
    # model_2B.fit(neural_X_train, neural_Y_train, epochs=70)
    # neural_pred_Y = model_2B.predict(np.asarray(X_test_neural).astype('float32'))
    # for i in range(len(neural_pred_Y)):
    #     for c in range(len(neural_pred_Y[i])):
    #         if neural_pred_Y[i][c] < 0.5:
    #             neural_pred_Y[i][c] = 0
    #         else:
    #             neural_pred_Y[i][c] =  1
    #
    # print(F"{accuracy_score(np.asarray(y_test_neural).astype('float32'), neural_pred_Y):.2%} - Neural Network")
    # <--------------------------------------------------------- Neural Network ------------------------------------->

    # <--------------------------------------------------------- Test of Different Models --------------------------->
    # classifiers = [BernoulliNB(),
    #                ComplementNB(),
    #                MultinomialNB(),
    #                KNeighborsClassifier(),
    #                DecisionTreeClassifier(),
    #                RandomForestClassifier(),
    #                LogisticRegression(),
    #                MLPClassifier(max_iter=1000),
    #                AdaBoostClassifier(),
    #                SVC()
    #                ]
    # print(type(classifiers))
    # for i in range(len(classifiers)):
    #     classifier = classifiers[i]
    #     classifier.fit(X_train, y_train)
    #     y_pred = classifier.predict(X_test)
    #     print(F"{accuracy_score(y_test, y_pred):.2%} - {i}")
    # <--------------------------------------------------------- Test of Different Models --------------------------->
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(F"{accuracy_score(y_test, y_pred):.2%} - Random Forest Model")


if '__main__' == __name__:
    restaurantReviews()
