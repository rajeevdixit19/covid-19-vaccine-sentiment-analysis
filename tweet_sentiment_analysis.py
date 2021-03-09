import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import csv_data_parser as cdp


if __name__ == '__main__':
    parser = cdp.DataParser('./dataset/vaccination_all_tweets.csv')
    tweets = parser.get_tweets()
    sia = SentimentIntensityAnalyzer()

    tfidf_vec = TfidfVectorizer(min_df=3, max_df=0.5, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                lowercase=True, ngram_range=(1, 1), stop_words='english')

    tfidf_mat = tfidf_vec.fit_transform(tweets['text'])

    df = pd.DataFrame(tfidf_mat.todense().tolist(), columns=tfidf_vec.get_feature_names())

    POSITIVE_WORD_FILE = './dataset/positive-words.txt'
    NEGATIVE_WORD_FILE = './dataset/negative-words.txt'

    f = open(POSITIVE_WORD_FILE, "r")
    positive_word_list = set(f.read().split('\n'))
    f.close()

    f = open(NEGATIVE_WORD_FILE, "r")
    negative_word_list = set(f.read().split('\n'))
    f.close()

    sentiment_col = pd.Series([], dtype=str)
    score_col = pd.Series([], dtype=float)
    nltk_sentiment_col = pd.Series([], dtype=str)
    nltk_score_col = pd.Series([], dtype=float)
    combined_sentiment_col = pd.Series([], dtype=str)
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



        score_col[i] = score
        if score > threshold_neut:
            sentiment_col[i] = "Positive"
        elif score < -1. * threshold_neut:
            sentiment_col[i] = "Negative"
        else:
            sentiment_col[i] = "Neutral"

        nltk_score = sia.polarity_scores(tweets.iloc[i][1])['compound']
        nltk_score_col[i] = nltk_score

        if nltk_score > threshold_neut:
            nltk_sentiment_col[i] = "Positive"
        elif nltk_score < -1. * threshold_neut:
            nltk_sentiment_col[i] = "Negative"
        else:
            nltk_sentiment_col[i] = "Neutral"

        combined_score = nltk_score * 0.5 + score * 0.5
        combined_score_col[i] = combined_score

        if combined_score > threshold_neut:
            combined_sentiment_col[i] = "Positive"
        elif combined_score < -1. * threshold_neut:
            combined_sentiment_col[i] = "Negative"
        else:
            combined_sentiment_col[i] = "Neutral"

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

    tweets['score'] = score_col
    tweets['sentiment'] = sentiment_col
    tweets['nltk_score'] = nltk_score_col
    tweets['nltk_sentiment'] = nltk_sentiment_col
    tweets['combined_score'] = combined_score_col
    tweets['combined_sentiment'] = combined_sentiment_col
    tweets['combined_granular_sentiment'] = combined_granular_sentiment_col

    tweets.to_csv('./dataset/tweet_sentiment_all_001.csv')

    agg_tfidf_frame = tweets.groupby(['sentiment']).count().iloc[:, 0].to_frame()
    agg_tfidf_frame.to_csv('./dataset/aggregations_all/tfidf_agg.csv')

    agg_nltk_frame = tweets.groupby(['nltk_sentiment']).count().iloc[:, 0].to_frame()
    agg_nltk_frame.to_csv('./dataset/aggregations_all/nltk_agg.csv')

    agg_tot_frame = tweets.groupby(['combined_sentiment']).count().iloc[:, 0].to_frame()
    agg_tot_frame.to_csv('./dataset/aggregations_all/tot_agg.csv')

    agg_gran_frame = tweets.groupby(['combined_granular_sentiment']).count().iloc[:, 0].to_frame()
    agg_gran_frame.to_csv('./dataset/aggregations_all/gran_agg.csv')

    # print(tweets)
