# run topic modeling with Bertopic
# 64GB memory needed. GPU highly recommended
# install CUML: pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==25.2.*"

import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import torch
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='politics')
    parser.add_argument('--min_cluster_size', type=int, default=40)
    parser.add_argument('--embedding_model_id', type=str, default='all-mpnet-base-v2')
    args = parser.parse_args()

    domain = args.domain
    min_cluster_size = args.min_cluster_size
    embedding_model_id = args.embedding_model_id

    print(domain)
    df = pd.read_csv(f'../1_reddit_data_processing/data/{domain}-all.csv')
    docs = df['cleaned_text'].tolist()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    sentence_model = SentenceTransformer(embedding_model_id, device=device)
    embeddings = np.load(f'../1_reddit_data_processing/embeds/embeds-{domain}.npy')

    from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
    representation_model = MaximalMarginalRelevance(diversity=0.3)
    # representation_model = KeyBERTInspired()

    from cuml.manifold import UMAP
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=2024)

    from cuml.cluster import HDBSCAN
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom',
                            prediction_data=True)

    topic_model = BERTopic(embedding_model=sentence_model,
                           representation_model=representation_model,
                           umap_model=umap_model,
                           hdbscan_model=hdbscan_model)
    print('Fitting topic model....')
    topics, probs = topic_model.fit_transform(docs, embeddings)
    print('Done\n')
    df['topic'] = topics
    df = df.query("topic!=-1")

    os.makedirs('topic_data', exist_ok=True)
    df.to_csv(f'topic_data/df-{domain}.csv', index=False)

    t = topic_model.get_topic_info()
    t.to_csv(f'topic_data/df-topics-{domain}.csv', index=False)

    os.makedirs(f'topic_models/{domain}', exist_ok=True)
    topic_model.save(f'topic_models/{domain}', serialization="safetensors", save_ctfidf=True,
                     save_embedding_model=f'sentence-transformers/{embedding_model_id}')