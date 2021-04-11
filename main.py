import nltk
from pprint import pprint


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


if __name__ == '__main__':
    myFunction()