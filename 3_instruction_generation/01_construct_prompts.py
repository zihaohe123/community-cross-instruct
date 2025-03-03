# construct prompts that are used to query the advanced LLM (e.g., GPT-4o)

import os
import pandas as pd
import numpy as np
import argparse

domain2reddit = {
    'politics': ['Anarcho_Capitalism', 'Liberal', 'NeutralPolitics', 'Conservative', 'AskThe_Donald'],
    'fitness': ['keto', 'WeightLossAdvice', 'EDAnonymous']
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='politics')
    parser.add_argument('--max_comment_len', type=int, default=256)
    args = parser.parse_args()

    domain = args.domain
    subreddits = domain2reddit[domain]


    def format_prompt_multi_subreddits(n_comments_subreddits, comments_subreddits, topic_keywords):
        n_comments_subreddits = np.array(n_comments_subreddits)
        n_nonzero_subreddits = (n_comments_subreddits != 0).sum()

        if n_nonzero_subreddits > 1:
            s = f"Below are comments from {n_nonzero_subreddits} different subreddits related to a topic. The topic can be represented using these keywords: {topic_keywords}.\n\n"
        else:
            s = f"Below are comments from a subreddit related to a topic. The topic can be represented using these keywords: {topic_keywords}.\n\n"

        for i, subreddit in enumerate(subreddits):
            n_comments = n_comments_subreddits[i]
            if n_comments == 0:
                continue
            comments = comments_subreddits[i]
            comments = comments.split('||\n')
            s_comments = f'Comments from r/{subreddit}\n'
            for j, comment in enumerate(comments):
                comment = ' '.join(comment.split()[:args.max_comment_len])
                s_comments += f'Comment {j + 1}: {comment}\n'
            s_comments += '\n'
            s += s_comments

        if n_nonzero_subreddits > 1:
            # s_instruction = f"""Write 5 questions (Q1 through Q5) on this topic that can be answered based on these comments. For a subreddit, each question should be answered in a way that the members from the subreddit would do, and the answers should echo the comments shown above. Do NOT rely on your background knowledge about the specific subreddits to answer the questions. The questions should be low-level, detailed, and trigger different responses that differentiate between different subreddits. Don’t ask too high-level questions. The questions should not be in the style of reading comprehension ones, and they are intended for members in the subreddits to answer. The questions should not contain “comment” in them. Each question should be paired with answers from all {n_nonzero_subreddits} subreddits. For the first 3 questions, they are open-ended. The answers should be concise (fewer than 32 tokens), legible, grammatically correct. For the second 2 questions, they are multi-choice questions and are associated with four options (A through D). Below is the format of generated questions.\n\n"""
            s_instruction = f"""Write 5 questions (Q1 through Q5) on this topic that can be answered based on these comments. For a subreddit, each question should be answered in a way that the members from the subreddit would do, and the answers should echo the comments shown above. Do NOT rely on your background knowledge about the specific subreddits to answer the questions. The questions should be low-level, detailed. Don’t ask too high-level questions. The questions should not be in the style of reading comprehension ones, and they are intended for members in the subreddits to answer. The questions should not contain “comment” in them. Each question should be paired with answers from all {n_nonzero_subreddits} subreddits. For the first 3 questions, they are open-ended. The answers should be concise (fewer than 32 tokens), legible, grammatically correct. For the second 2 questions, they are multi-choice questions and are associated with four options (A through D). Try to come up questions that members from different subreddits would answer differently. Below is the format of generated questions.\n\n"""
        else:
            s_instruction = f"""Write 5 questions (Q1 through Q5) on this topic that can be answered based on these comments. For the subreddit, each question should be answered in a way that the members from the subreddit would do, and the answers should echo the comments shown above. Do NOT rely on your background knowledge about the specific subreddit to answer the questions. The questions should be low-level and detailed. Don’t ask too high-level questions. The questions should not be in the style of reading comprehension ones, and they are intended for members in the subreddits to answer. Each question should be paired with answers from this subreddit. For the first 3 questions, they are open-ended. The answers should be concise (fewer than 32 tokens), legible, grammatically correct. For the second 2 questions, they are multi-choice questions and are associated with four options (A through D). x`Below is the format of generated questions.\n\n"""

        s_template_open = "Open-ended Questions\nQ1: [open-ended question]\n"
        for i, subreddit in enumerate(subreddits):
            n_comments = n_comments_subreddits[i]
            if n_comments == 0:
                continue
            s_template_open += f'Answer from {subreddit} {i}: [answer in clean text]\n'
        s_template_open += '\n'

        s_template_open += '....\n\n'

        s_template_open2 = "Q3: [open-ended question]\n"
        for i, subreddit in enumerate(subreddits):
            n_comments = n_comments_subreddits[i]
            if n_comments == 0:
                continue
            s_template_open2 += f'Answer from {subreddit} {i}: [answer in clean text]\n'
        s_template_open2 += '\n\n'

        s_template_closed = "Closed-ended Questions\nQ4: [multi-choice question]\nA.xxx\nB.xxx\nC.xxx\nD.xxx\n"
        for i, subreddit in enumerate(subreddits):
            n_comments = n_comments_subreddits[i]
            if n_comments == 0:
                continue
            s_template_closed += f'Answer from {subreddit} {i}: A/B/C/D\n'
        s_template_closed += '\n'

        # s_template_closed += '....\n\n'

        s_template_closed2 = "Q5: [multi-choice question]\nA.xxx\nB.xxx\nC.xxx\nD.xxx\n"
        for i, subreddit in enumerate(subreddits):
            n_comments = n_comments_subreddits[i]
            if n_comments == 0:
                continue
            s_template_closed2 += f'Answer from {subreddit} {i}: A/B/C/D\n'
        s_template_closed2 += '\n'

        s += s_instruction + s_template_open + s_template_open2 + s_template_closed + s_template_closed2

        return s

    def apply_format_template(row):
        keywords = row['keywords']
        n_comments_subreddits = []
        comments_subreddits = []
        for subreddit in subreddits:
            n_comments_subreddits.append(row[f'n_{subreddit}_texts'])
            comments_subreddits.append(row[f'{subreddit}_texts'])
        s = format_prompt_multi_subreddits(n_comments_subreddits, comments_subreddits, keywords)
        s = s.strip('\n')
        return s

    df = pd.read_csv(f'../2_topic_modeling/data_for_query/df-{domain}-for-query.csv')
    prompts = df.apply(apply_format_template, axis=1).tolist()
    df_prompts = pd.DataFrame(prompts, columns=['prompt'])
    df_prompts.to_csv(f'df_prompts_{domain}.csv', index=False)
    print(df_prompts.shape)

    os.makedirs(f'prompts/{domain}', exist_ok=True)
    for i, prompt in enumerate(prompts):
        with open(f'prompts/{domain}/{i+1}.txt', 'w') as f:
            f.write(prompts[i])