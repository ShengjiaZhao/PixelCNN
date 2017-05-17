from sklearn import svm
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from abstract_network import *


# Import data
data_path = 'data'
if not os.path.exists(data_path):
    os.makedirs(data_path)
mnist = input_data.read_data_sets(data_path)


batches = []
labels = []
for _ in range(10):
    batch_x, batch_y = mnist.train.next_batch(100)
    batch_x *= 200.0
    batches.append(batch_x)
    labels.append(batch_y)
batches = np.concatenate(batches, axis=0)
labels = np.concatenate(labels)

# Compute pair-wise distance
x_range = np.sqrt(np.sum(np.square(np.max(batches, axis=0) - np.min(batches, axis=0))))
gamma = 0.0001 / x_range
while True:
    classifier = svm.SVC(decision_function_shape='ovr', gamma=gamma)
    classifier.fit(batches, labels)

    correct_count = 0
    for i in range(10):
        batch_x, batch_y = mnist.test.next_batch(100)
        batch_x *= 200.0
        pred = classifier.predict(batch_x)
        correct_count += np.sum([1 for j in range(100) if batch_y[j] == pred[j]])
    print("%f %d" % (gamma, correct_count))
    gamma *= 2.0
    if gamma > 100.0:
        break