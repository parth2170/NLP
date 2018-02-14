from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import Counter
import math
import os
import random
from tempfile import gettempdir
import zipfile
import string
from nltk import word_tokenize
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
import csv
# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
  local_filename = os.path.join(gettempdir(), filename)
  if not os.path.exists(local_filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
  statinfo = os.stat(local_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')
  return local_filename


filename = maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#       This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
#data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
                                                            
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)


new_data0=[]
with open('Reviews.csv','r') as file:
    read=csv.reader(file, delimiter=',')
    try:
        for row in read:
            word_tokens = word_tokenize(row[8])
            for w in word_tokens:
                if (w not in stop_words) and (w not in string.punctuation):
                    new_data0.append(w.lower())
    except ValueError:
        print("Some Data Ignored")
new_data_dict=Counter(new_data0)
new_data_size=len(new_data_dict)

new_data, new_count, new_dictionary, new_reverse_dictionary = build_dataset(new_data0, new_data_size)

new_vocabulary_size=len(new_dictionary)

'''takes a list of tuples x and converts it into a dictionary y with normalized frequencies'''
def dictmaker(x):
    y={}
    for i in range(len(x)):
        y[x[i][0]]=x[i][1]/x[21][1]
    return y

source_freq_dict=dictmaker(count)
target_freq_dict=dictmaker(new_count)

#takes input the normalized frequency of two words and returns the significance function
def signify(x,y):
    return 2*x*y/(x+y)

#calculatess sigmoid of x
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
'''finding common words'''

common={}
common_use={}
set1=set(dictionary)
set2=set(new_dictionary)
for e in set1.intersection(set2):
    common[e]=dictionary[e]
    common_use[e]=new_dictionary[e]
    
'''loading pre trained source embeddings'''

def floater(x):
    z=[]
    for i in x:
        z.append(np.float32(i))
    return z
source_embeddings=[]
with open("source_vec.csv", 'r') as file:
    read=csv.reader(file, delimiter=',')
    for row in read:
        if row!=[]:
            source_embeddings.append(floater(row))
source_embeddings=np.array(source_embeddings)

'''storing the common source embeddings into an array of shape same as the target embeddings
    the non common words have 0 vector as their source embeddings'''
    
'''also calculating alpha'''

alpha=np.zeros(new_vocabulary_size)
lambdaa=1

common_source_embeddings=np.zeros(shape=(new_vocabulary_size, 128))
for i in range(new_vocabulary_size):
    for j in common_use:
        if i==common_use[j]:
            common_source_embeddings[i]=source_embeddings[common[j]]
            alpha[i]=sigmoid(lambdaa*signify(source_freq_dict[j], target_freq_dict[j]))

alpha=np.sqrt(alpha)
#del vocabulary
del new_data0  # Hint to reduce memory.
#print('Most common words (+UNK)', count[:5])
#print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

print('Most common words (+UNK)', new_count[:5])
print('Sample data', new_data[:10], [new_reverse_dictionary[i] for i in new_data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window, data):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      print(len(data))
      #buffer[:] = data[:span]
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

#batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, data=data)
data_index = 0
new_batch, new_labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, data=new_data)

#for i in range(8):
  #print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
  
for i in range(8):
  print(new_batch[i], new_reverse_dictionary[new_batch[i]], '->', new_labels[i, 0], new_reverse_dictionary[new_labels[i, 0]])


# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()
with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        with tf.device('/cpu:0'):
            '''Converting common source embeddings to a tensorflow constant'''
            common_source_embeddings=tf.Variable(common_source_embeddings, dtype=tf.float32)
            #tf.cast(common_source_embeddings, tf.float64)
            #common_source_embeddings=tf.convert_to_tensor(common_source_embeddings, dtype=float32)

            embeddings = tf.Variable(tf.random_uniform([new_vocabulary_size, embedding_size], -1.0, 1.0))
            reg_embeddings=tf.subtract(embeddings, common_source_embeddings) 
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            nce_weights = tf.Variable(tf.truncated_normal([new_vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([new_vocabulary_size]))
        loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=new_vocabulary_size))
        '''Adding regularization term'''
        alpha=tf.Variable(alpha, dtype=tf.float32)
        for i in range(new_vocabulary_size):
            q=reg_embeddings[i]*alpha[i]
            z=tf.concat([z,q], 0)
        #reg_embeddings=tf.Variable(tf.convert_to_tensor(reg_embeddings),dtype=tf.float32)
        regularizer = tf.nn.l2_loss(reg_embeddings)
        loss = tf.reduce_mean(loss + 0.001*regularizer)

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
        init = tf.global_variables_initializer()

  # Add variable initializer.

# Step 5: Begin training.
num_steps = 50001

print("*********DONE*************")


with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')
  q=0
  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window, new_data)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = new_reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = new_reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [new_reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)

