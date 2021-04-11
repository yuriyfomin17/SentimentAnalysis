import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


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


def lemmatizeWords(final_words):
    wn = WordNetLemmatizer()
    lem_words = []
    for word in final_words:
        word = wn.lemmatize(word)
        lem_words.append(word)
    # The frequency distribution of the words
    freq_dis = nltk.FreqDist(lem_words)

    # Frequency Distribution Plot
    plt.subplots(figsize=(20, 12))
    freq_dis.plot(30)


def restaurantReviews():
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

    # dividing reviews into positive nad negative review
    positive_rev = dividing_reviews(1, dataset)

    negative_rev = dividing_reviews(0, dataset)

    final_words_pos = filtering_Reviews(positive_rev)
    final_words_neg = filtering_Reviews(negative_rev)
    lemmatizeWords(final_words_pos)


if '__main__' == __name__:
    restaurantReviews()
