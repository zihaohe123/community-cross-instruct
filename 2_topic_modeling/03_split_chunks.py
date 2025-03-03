# split the original docs into chunks

import os
import pandas as pd
import argparse
import ast
import numpy as np
import random
random.seed(2024)


def sample_and_remove_elements(elements_list, sample_size=20):
    if len(elements_list) < sample_size:
        sample_size = len(elements_list)  # Adjust sample size if list has fewer elements
    sample = random.sample(elements_list, sample_size)
    for element in sample:
        elements_list.remove(element)
    return sample, elements_list

# politics -- 72 topics
# fitness -- 148 topics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='politics')
    parser.add_argument('--max_n_docs', type=int, default=50, help='max # of docs per chunk')
    parser.add_argument('--min_n_docs', type=int, default=30, help='min # of docs per chunk')
    parser.add_argument('--max_n_chunks', type=int, default=5, help='max # of chunks per topic')
    parser.add_argument('--min_n_subreddits', type=int, default=4, help='min # of subreddits per chunk')
    parser.add_argument('--test_ratio', type=int, default=0.15, help='ratio of the test data')
    args = parser.parse_args()

    domain2reddit = {
        'politics': ['Anarcho_Capitalism', 'Liberal', 'NeutralPolitics', 'Conservative', 'AskThe_Donald'],
        'fitness': ['keto', 'WeightLossAdvice', 'EDAnonymous']
    }

    domain = args.domain
    subreddits = domain2reddit[domain]
    print(domain, subreddits)

    df_data = pd.read_csv(f'topic_data/df-{domain}.csv')
    df_topics = pd.read_csv(f'topic_data/df-topics-{domain}.csv')
    df_topics['Representation'] = df_topics['Representation'].apply(ast.literal_eval)

    topics = df_topics['Topic'].tolist()
    topics.remove(-1)
    # topics = topics[:args.max_n_topics]
    topic2keywords = df_topics.set_index('Topic')['Representation']

    data = []
    # corpus = []
    for topic in topics:
        print(topic)
        df_data_topic = df_data.query(f"topic=={topic}")
        subreddit2texts = {}
        for subreddit in subreddits:
            texts = df_data_topic.query(f"subreddit=='{subreddit}'")['cleaned_text'].tolist()
            n_texts = len(texts)
            n_samples = min(n_texts, args.max_n_docs * args.max_n_chunks)
            subreddit2texts[subreddit] = random.sample(texts, n_samples)
        keywords = ','.join(topic2keywords[topic])

        k = 0
        while True:
            print(topic, k)
            k += 1
            row_texts = []
            row_texts_count = []
            for subreddit in subreddits:
                texts = subreddit2texts[subreddit]
                sampled_texts, new_texts = sample_and_remove_elements(texts, args.max_n_docs)
                n_sampled = len(sampled_texts)
                # for i in range(n_sampled):
                #     corpus.append([topic, keywords, subreddit, sampled_texts[i]])
                sampled_text_str = '||\n'.join(sampled_texts) if n_sampled > 0 else ''
                subreddit2texts[subreddit] = new_texts
                if n_sampled < args.min_n_docs:
                    sampled_text_str = ''
                    n_sampled = 0
                row_texts.append(sampled_text_str)
                row_texts_count.append(n_sampled)

            if sum(row_texts_count) == 0:
                break

            row = [topic, keywords]
            for i in range(len(subreddits)):
                row.append(row_texts[i])
                row.append(row_texts_count[i])
            data.append(row)

    columns = ['topic', 'keywords']
    for subreddit in subreddits:
        columns.append(f'{subreddit}_texts')
        columns.append(f'n_{subreddit}_texts')

    # normally split
    df_data_for_query = pd.DataFrame(data, columns=columns)
    # df_corpus = pd.DataFrame(corpus, columns=['topic', 'keywords', 'subreddit', 'text'])

    def count_subreddits(row):
        n = 0
        for subreddit in subreddits:
            n += int(row[f'n_{subreddit}_texts'] != 0)
        return n

    df_data_for_query = df_data_for_query.sample(frac=1).reset_index(drop=True)
    df_data_for_query['n_subreddits'] = df_data_for_query.apply(count_subreddits, axis=1)
    df_data_for_query = df_data_for_query.sort_values(by='n_subreddits', ascending=False)
    # df_data_for_query = df_data_for_query[:df_data_for_query.shape[0]//2]
    df_data_for_query = df_data_for_query.query(f"n_subreddits>={args.min_n_subreddits}")


    df_data_for_query['id'] = list(range(df_data_for_query.shape[0]))
    splits = ['train'] * df_data_for_query.shape[0]
    for i in range(int(args.test_ratio*df_data_for_query.shape[0])):
        splits[i] = 'test'
    random.shuffle(splits)
    df_data_for_query['split'] = splits

    # split by topics
    topics = df_data_for_query['topic'].unique().tolist()
    print(f"******************{len(topics)}******************")
    test_splits_by_topic = topics
    # for i in range(10):
    #     test_splits_by_topic.remove(i)
    test_splits_by_topic = random.sample(test_splits_by_topic,
                                         int(args.test_ratio * len(topics)))



    df_data_for_query['split_by_topic'] = df_data_for_query['topic'].apply(
        lambda x: 'train' if x not in test_splits_by_topic else 'test')

    # df_data_for_query = df_data_for_query.sort_values(by='topic')

    os.makedirs('data_for_query', exist_ok=True)
    df_data_for_query.to_csv(f'data_for_query/df-{domain}-for-query.csv', index=False)
