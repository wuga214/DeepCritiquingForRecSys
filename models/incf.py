import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from scipy.sparse import vstack, hstack


class INCF(object):
    def __init__(self,
                 label_dim,
                 text_dim,
                 embed_dim,
                 num_layers,
                 batch_size,
                 lamb=0.01,
                 learning_rate=1e-3,
                 optimizer=tf.train.AdamOptimizer,
                 **unused):
        self.label_dim = label_dim
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):
        pass
