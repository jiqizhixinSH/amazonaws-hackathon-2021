"""
Copyright: Qinzi Now, Tencent Cloud.
"""
import sys
sys.path.append("..")
sys.path.append("../..")
import pandas as pd
import text2vec
from text2vec import Similarity
from text2vec import SearchSimilarity
import numpy as np


def main(input_word):
    # input_embed = text2vec.encode(input_word)
    data = pd.read_csv('../raw_data/test_samples.csv', encoding='gbk')
    corpus = data['0'].to_numpy()
    if input_word in corpus:
        # input_index = corpus.index(input_word)
        input_index = np.where(corpus==input_word)
        # print(int(input_index[0]))
        return data.loc[int(input_index[0]), '1']
    search_sim = SearchSimilarity(corpus=corpus)
    scores = search_sim.get_scores(query=input_word)
    max_index = np.where(scores==np.max(scores))
    return data.loc[max_index, '1']


if __name__ == '__main__':
    print(main('餐桌前'))
