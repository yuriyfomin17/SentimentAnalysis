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
    # plt.subplots(figsize=(20, 12))
    # freq_dist.plot(30)
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

    # Creating the Bag of Words model
    cv = CountVectorizer(max_features=2000)
    corpus = []
    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for i in range(0, 1000): #as the data as 1000 data points
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    cv = CountVectorizer(max_features = 2000)
    #the X and y
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values
    print("Hello")


if '__main__' == __name__:
    restaurantReviews()
