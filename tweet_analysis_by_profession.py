import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import csv_data_parser as cdp


if __name__ == '__main__':
    parser = cdp.DataParser('./dataset/vaccination_all_tweets.csv')
    tweets = parser.get_tweets()

    parser = cdp.DataParser('./dataset/healthcare_workers.csv')
    healthcare_tweets = parser.get_tweets()

    parser = cdp.DataParser('./dataset/media_corporations.csv')
    media_corp_tweets = parser.get_tweets()

    parser = cdp.DataParser('./dataset/media_users.csv')
    media_user_tweets = parser.get_tweets()

    sia = SentimentIntensityAnalyzer()

    tfidf_vec = TfidfVectorizer(min_df=3, max_df=0.5, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                lowercase=True, ngram_range=(1, 1), stop_words='english')

    tfidf_vec = tfidf_vec.fit(tweets['text'])

    tfidf_mat = tfidf_vec.transform(media_corp_tweets['text'])

    df = pd.DataFrame(tfidf_mat.todense().tolist(), columns=tfidf_vec.get_feature_names())

    POSITIVE_WORD_FILE = './dataset/positive-words.txt'
    NEGATIVE_WORD_FILE = './dataset/negative-words.txt'

    f = open(POSITIVE_WORD_FILE, "r")
    positive_word_list = set(f.read().split('\n'))
    f.close()

    f = open(NEGATIVE_WORD_FILE, "r")
    negative_word_list = set(f.read().split('\n'))
    f.close()


    combined_granular_sentiment_col = pd.Series([], dtype=str)
    combined_score_col = pd.Series([], dtype=float)

    threshold_neut = 0.1
    threshold_mild = 0.4

    for i in range(df.shape[0]):
        if i % 5 == 0:
            print(i)
        score = 0.0
        for j in range(df.shape[1]):
            if str(df.columns[j]) in positive_word_list:
                score += df.iloc[i][j]
            elif str(df.columns[j]) in negative_word_list:
                score -= df.iloc[i][j]

        nltk_score = sia.polarity_scores(media_corp_tweets.iloc[i][1])['compound']

        combined_score = nltk_score * 0.5 + score * 0.5
        combined_score_col[i] = combined_score

        if combined_score < -1. * threshold_mild:
            combined_granular_sentiment_col[i] = 'Negative'
        elif combined_score < -1. * threshold_neut:
            combined_granular_sentiment_col[i] = 'Skeptical'
        elif combined_score < threshold_neut:
            combined_granular_sentiment_col[i] = 'Neutral'
        elif combined_score < threshold_mild:
            combined_granular_sentiment_col[i] = 'Optimistic'
        else:
            combined_granular_sentiment_col[i] = 'Positive'

    media_corp_tweets['combined_score'] = combined_score_col
    media_corp_tweets['combined_granular_sentiment'] = combined_granular_sentiment_col

    media_corp_tweets.to_csv('./dataset/media_corp_001.csv')

    # agg_gran_frame = media_corp_tweets.groupby(['combined_granular_sentiment']).count().iloc[:, 0].to_frame()
    # agg_gran_frame.to_csv('./dataset/aggregations_prof/media_corp_agg.csv')

    # print(tweets)
