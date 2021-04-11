import pandas as pd
import string


def dividing_reviews(parameter, dataset):
    reviews = []
    for i in range(len(dataset.Review.values)):
        if dataset.Liked.values[i] == parameter:
            reviews.append(dataset.Review.values[i])
    return reviews


def restaurantReviews():
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
    positive_review = dividing_reviews(1, dataset)

    negative_review_ids = dividing_reviews(0, dataset)
    print('hello')


if '__main__' == __name__:
    restaurantReviews()
