import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


def restaurantReviewsTest():
    # model filename
    classifier = 'trained_restaurant_reviews_classifier.sav'
    loaded_classifier = pickle.load(open(classifier, 'rb'))

    with open('x_test.data', 'rb') as filehandle:
        # read the data as binary data stream
        x_test = pickle.load(filehandle)

    with open('y_test.data', 'rb') as filehandle:
        # read the data as binary data stream
        y_test = pickle.load(filehandle)

    y_pred = loaded_classifier.predict(x_test)

    print(F"{accuracy_score(y_test, y_pred):.2%} - Random Forest Model")


if '__main__' == __name__:
    restaurantReviewsTest()
