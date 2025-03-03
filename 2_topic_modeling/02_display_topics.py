# display the topic statistics

import os
import pandas as pd
import argparse
import ast
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='politics')
    args = parser.parse_args()

    domain2reddit = {
        'politics': ['Anarcho_Capitalism', 'Liberal', 'NeutralPolitics', 'Conservative', 'AskThe_Donald'],
        'fitness': ['keto', 'WeightLossAdvice', 'EDAnonymous']
    }


    domain = args.domain
    subreddits = domain2reddit[domain]
    print(domain, subreddits)

    df_data = pd.read_csv(f'topic_data/df-{domain}.csv')
    topics = df_data['topic'].unique()
    data = []
    for topic in topics:
        total = df_data.query('topic == @topic').shape[0]
        row = [topic, total]
        for subreddit in subreddits:
            count = df_data.query('topic == @topic and subreddit == @subreddit').shape[0]
            row.append(count)
        data.append(row)
    cols = ['topic', 'total'] + subreddits
    df = pd.DataFrame(data, columns=cols)
    df = df.sort_values(by=['topic'], ascending=True)
    df.to_csv(f'topic_data/df-topic-stats-{domain}.csv', index=False)