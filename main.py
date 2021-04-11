import nltk
from pprint import pprint
from nltk.sentiment import SentimentIntensityAnalyzer
from random import shuffle
from statistics import mean

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

classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "AdaBoostClassifier": AdaBoostClassifier(),
}


def is_positive(tweet: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise"""
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(tweet)['compound'] > 0


def tweetsExample():
    tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]
    shuffle(tweets)
    for tweet in tweets[:10]:
        print(">", is_positive(tweet), tweet)


def myFunction():
    # <--- General Info --->
    # text = nltk.Text(nltk.corpus.state_union.words())
    # stopwords = nltk.corpus.stopwords.words("english")
    # words = [w for w in text if w.lower() not in stopwords and w.isalpha()]
    # fd = nltk.FreqDist([w.lower() for w in words])
    # print(fd.most_common(3))

    # <--- Method with will be useful for my case ---->
    # text = """Beautiful is better than ugly. Explicit is better than implicit.Simple is better than complex."""
    # words: list[str] = nltk.word_tokenize(text)
    # stopwords = nltk.corpus.stopwords.words("english")
    # words = [w for w in words if w.lower() not in stopwords and w.isalpha()]
    # text = nltk.Text(words)
    # fd = text.vocab()
    # print(fd.most_common(3))

    # <--- Method for finding collocations --->
    # <--- collocations are series of words that appear together often. Bigrams, Trigrams, Quadgrams --->
    # stopwords = nltk.corpus.stopwords.words("english")
    # words = [w.lower() for w in nltk.corpus.state_union.words() if w not in stopwords and w.isalpha()]
    # finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
    # print(finder.ngram_fd.most_common(4))

    # <--- Example usage of VADER - pretrained sentiment analyzer --->
    # sia = SentimentIntensityAnalyzer()
    # result = sia.polarity_scores("Wow, NLTK is really powerful!")
    # print(result)
    print("hello")


def isPositiveMovieReview(review_id: str) -> bool:
    """True if the average of all sentence compound scores is positive."""
    text = nltk.corpus.movie_reviews.raw(review_id)
    sia = SentimentIntensityAnalyzer()
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(text)
    ]
    return mean(scores) > 0


def movieReviewsComplexFeature():

    def skip_unwanted(pos_tuple):
        word, tag = pos_tuple
        if not word.isalpha() or word in unwanted:
            return False
        if tag.startswith("NN"):
            return False
        return True

    unwanted = nltk.corpus.stopwords.words('english')
    unwanted.extend([w.lower for w in nltk.corpus.names.words()])
    positive_words = [word for word, tag in filter(
        skip_unwanted,
        nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
    )]
    negative_words = [word for word, tag in filter(
        skip_unwanted,
        nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
    )]

    # Option to add Bigram Collocation feature to the existing dataset
    positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
        w for w in nltk.corpus.movie_reviews.words(categories=["pos"])
        if w.isalpha() and w not in unwanted
    ])
    negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
        w for w in nltk.corpus.movie_reviews.words(categories=["neg"])
        if w.isalpha() and w not in unwanted
    ])

    # Delete common words occurring in intersection
    positive_fd = nltk.FreqDist(positive_words)
    negative_fd = nltk.FreqDist(negative_words)
    common_set = set(positive_fd).intersection(negative_fd)
    for word in common_set:
        del positive_fd[word]
        del negative_fd[word]

    # Get top 100 positive and top 100 negative words in reviews
    top_100_positive = {word for word, count in positive_fd.most_common(100)}
    top_100_negative = {word for word, count in negative_fd.most_common(100)}

    def extract_features(text):
        curr_features = dict()
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
                # if word in positive_bigram_finder:
                #     bigram_count_pos += 1
                # if word in negative_bigram_finder:
                #     bigram_count_neg += 1
            compound_scores.append(sia.polarity_scores(sentence)["compound"])
            positive_scores.append(sia.polarity_scores(sentence)["pos"])

        # Adding 1 to the final compound score to always have positive numbers
        # since some classifiers you'll use later don't work with negative numbers.
        curr_features["mean_compound"] = mean(compound_scores) + 1
        curr_features["mean_positive"] = mean(positive_scores)
        curr_features["wordcount_count_positive"] = wordcount_pos
        curr_features["wordcount_count_negative"] = wordcount_neg
        # curr_features["bigram_count_pos"] = bigram_count_pos
        # curr_features["bigram_count_neg"] = bigram_count_neg
        return curr_features

    features = [
        (extract_features(nltk.corpus.movie_reviews.raw(review)), "pos")
        for review in nltk.corpus.movie_reviews.fileids(categories=["pos"])
    ]
    features.extend([
        (extract_features(nltk.corpus.movie_reviews.raw(review)), "neg")
        for review in nltk.corpus.movie_reviews.fileids(categories=["neg"])
    ])

    train_count = len(features) // 4
    shuffle(features)
    classifier = nltk.NaiveBayesClassifier.train(features[:train_count])
    print(classifier.show_most_informative_features(10))
    for name, sklearn_classifier in classifiers.items():
        classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
        classifier.train(features[:train_count])
        accuracy = nltk.classify.accuracy(classifier, features[train_count:])
        print(F"{accuracy:.2%} - {name}")


def movieReviewSimple():
    positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
    negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])
    all_review_ids = positive_review_ids + negative_review_ids

    shuffle(all_review_ids)
    correct = 0
    for review_id in all_review_ids:
        if isPositiveMovieReview(review_id):
            if review_id in positive_review_ids:
                correct += 1
        else:
            if review_id in negative_review_ids:
                correct += 1
    print(F"{correct / len(all_review_ids):.2%} correct")


if __name__ == '__main__':
    movieReviewsComplexFeature()
