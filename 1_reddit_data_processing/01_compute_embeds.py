# compute the embeddings of the data, for the subsequent topic modeling

from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import torch
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='politics')
    args = parser.parse_args()
    domain = args.domain

    df = pd.read_csv(f'data/{domain}-all.csv')

    docs = df['cleaned_text'].tolist()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    os.makedirs('embeds', exist_ok=True)
    embed_path = f'embeds/embeds-{domain}.npy'
    sentence_model = SentenceTransformer('all-mpnet-base-v2', device=device)
    embeddings = sentence_model.encode(docs, show_progress_bar=True, batch_size=1600)

    print('saving embeddings....')
    np.save(embed_path, embeddings)
    print('Done')