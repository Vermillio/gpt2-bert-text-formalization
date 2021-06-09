"""
semantic critic
"""

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

def save_word_embeddings(vocab, vector_size, word_embed_filename):
    model = Word2Vec(vocab, epochs=100)
    model.wv.save_word2vec_format(word_embed_filename)

class SemanticCritic:

    def __init__(self, word_embed_filename):
        self.model = KeyedVectors.load_word2vec_format(word_embed_filename)

    def get_rewards(self, src, target):
        reward_list = []
        for i,j in zip(src, target):
            dist = min(100, self.model.wmdistance(i, j))
            reward = -1.0 * dist / float(len(i))
            reward_list.append(reward)
        return reward_list
