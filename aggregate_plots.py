import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # df = pd.read_csv('./dataset/aggregations_all/tot_agg.csv')
    df = pd.read_csv('./dataset/aggregations_all/gran_agg.csv')

    df['combined_granular_sentiment'] = pd.Categorical(df['combined_granular_sentiment'],
                                                              categories=['Negative', 'Skeptical', 'Neutral',
                                                                          'Optimistic', 'Positive'],
                                                              ordered=True)
    df = df.sort_values('combined_granular_sentiment')

    fig1, ax1 = plt.subplots()

    # col = ['#ff5555', '#66b3ff', '#88ff88']
    col = ['#ff5555', '#ffaa00', '#66b3ff', '#33ffdd', '#33ff33']

    # ax1.pie(df['id'], labels=df['combined_sentiment'], colors=col, autopct='%1.1f%%', startangle=90)
    ax1.pie(df['id'], labels=df['combined_granular_sentiment'], colors=col, autopct='%1.1f%%', startangle=90)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax1.axis('equal')
    plt.tight_layout()
    # plt.title('Vaccine Sentiment across all tweets')

    # fig1.savefig('./dataset/plots/sentiment_agg_pie_001.png')
    fig1.savefig('./dataset/plots/sentiment_gran_pie_001.png')

    plt.close(fig1)