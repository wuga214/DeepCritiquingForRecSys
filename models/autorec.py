import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from scipy.sparse import vstack, hstack
from utils.regularizers import Regularizer


class AutoRec(object):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 batch_size,
                 lamb=0.01,
                 learning_rate=1e-4,
                 optimizer=tf.train.RMSPropOptimizer,
                 **unused):
        self.input_dim = self.output_dim = input_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):
        self.inputs = tf.placeholder(tf.float32, (None, self.input_dim))

        with tf.variable_scope('encode'):
            encode_weights = tf.Variable(tf.truncated_normal([self.input_dim, self.embed_dim], stddev=1 / 500.0),
                                         name="Weights")
            encode_bias = tf.Variable(tf.constant(0., shape=[self.embed_dim]), name="Bias")

            self.encoded = tf.nn.relu(tf.matmul(self.inputs, encode_weights) + encode_bias)

        with tf.variable_scope('decode'):
            self.decode_weights = tf.Variable(tf.truncated_normal([self.embed_dim, self.output_dim], stddev=1 / 500.0),
                                     name="Weights")
            self.decode_bias = tf.Variable(tf.constant(0., shape=[self.output_dim]), name="Bias")
            prediction = tf.matmul(self.encoded, self.decode_weights) + self.decode_bias

            self.output = tf.nn.sigmoid(prediction)

        with tf.variable_scope('loss'):
            l2_loss = tf.nn.l2_loss(encode_weights) + tf.nn.l2_loss(self.decode_weights)
            sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.inputs, logits=prediction)
            self.loss = tf.reduce_mean(sigmoid_loss) + self.lamb*tf.reduce_mean(l2_loss)

        with tf.variable_scope('optimizer'):
            self.train = self.optimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def get_batches(self, rating_matrix, batch_size):
        remaining_size = rating_matrix.shape[0]
        batch_index = 0
        batches = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                batches.append(rating_matrix[batch_index*batch_size:])
            else:
                batches.append(rating_matrix[batch_index*batch_size:(batch_index+1)*batch_size])
            batch_index += 1
            remaining_size -= batch_size
        return batches

    def train_model(self, rating_matrix, epoch=100, batches=None, **unused):
        if batches is None:
            batches = self.get_batches(rating_matrix, self.batch_size)

        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step].todense()}
                training = self.sess.run([self.train], feed_dict=feed_dict)

    def predict(self, batch_matrix):
        feed_dict = {self.inputs: batch_matrix.todense()}
        return self.sess.run(self.output, feed_dict=feed_dict)
