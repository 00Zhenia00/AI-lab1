import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk import FreqDist, download as nltk_download
from nltk.tokenize import word_tokenize

def pd_series_to_wordcloud(column: pd.Series, stopwords: list):
    # text = " ".join(review for review in column)
    text = column.str.cat(sep=" ")
    plt.figure(figsize=(10, 5))
    wordcloud = WordCloud(stopwords=stopwords).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("Most frequently used words in series")
    plt.axis("off")
    plt.show()

def get_most_common_words(series: pd.Series, number, exceptions):
    nltk_download('punkt_tab', quiet=True)

    series = series.apply(lambda x: x.lower() if isinstance(x, str) else x)
    series = series.replace(to_replace=r'[^\w\s]', value='', regex=True)
    series = series.replace(to_replace=r'\d', value='', regex=True)

    text = series.str.cat(sep=" ")

    all_words = word_tokenize(text, language="english")
    all_words_except_stopwords = FreqDist(w.lower() for w in all_words if w.lower() not in exceptions)

    return list(dict(all_words_except_stopwords.most_common(number)).keys())
