import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    cat = ['Negative', 'Skeptical', 'Neutral', 'Optimistic', 'Positive']

    df_health = pd.read_csv('./dataset/aggregations_prof/healthcare_agg.csv')
    df_health['combined_granular_sentiment'] = pd.Categorical(df_health['combined_granular_sentiment'],
                                                              categories=cat, ordered=True)
    df_health = df_health.sort_values('combined_granular_sentiment')
    sum = df_health['id'].sum()
    df_health['id'] = df_health['id'] * 100.0 / sum

    df_media_corp = pd.read_csv('./dataset/aggregations_prof/media_corp_agg.csv')
    df_media_corp['combined_granular_sentiment'] = pd.Categorical(df_media_corp['combined_granular_sentiment'],
                                                                  categories=cat, ordered=True)
    df_media_corp = df_media_corp.sort_values('combined_granular_sentiment')
    sum = df_media_corp['id'].sum()
    df_media_corp['id'] = df_media_corp['id'] * 100.0 / sum

    df_media_user = pd.read_csv('./dataset/aggregations_prof/media_user_agg.csv')
    df_media_user['combined_granular_sentiment'] = pd.Categorical(df_media_user['combined_granular_sentiment'],
                                                                  categories=cat, ordered=True)
    df_media_user = df_media_user.sort_values('combined_granular_sentiment')
    sum = df_media_user['id'].sum()
    df_media_user['id'] = df_media_user['id'] * 100.0 / sum

    index = np.arange(5)
    width = 0.2

    plt.bar(index - width, df_health['id'], width, label='Healthcare Workers')
    plt.bar(index, df_media_user['id'], width, label='Media Personnel')
    plt.bar(index + width, df_media_corp['id'], width, label='Media Corporations')

    plt.ylabel('Percentage of tweets')
    plt.title('Sentiment Distribution by Occupation')

    plt.xticks(index + width / 2, cat)
    plt.legend(loc='best')

    plt.savefig('./dataset/plots/agg_category_bar.png')