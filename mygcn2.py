import numpy as np
import tensorflow as tf
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import squareform, pdist
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import copy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.layers import Dropout
from tensorflow.keras.backend import dropout


#----------------------------------------------------------------------
# Functions parse_index_file, sample_mask, sparse_to_tuple and
# load_file originate from Kipf's source code.
#----------------------------------------------------------------------
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data\ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data\ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess(adj, features):
    """
    :param adj:  adjacent matrix
    :param features:  samples
    :return:  normalized adjacent matrix and sample features
    """
    adj = adj.todense()
    adj = adj + np.eye(adj.shape[0])
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    A = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    # Row-normalize training samples
    features = features.todense()
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    # X = sparse_to_tuple(features)
    # X = tf.sparse.to_dense(features)

    return A, features


def gen_data():
    """
    Generate a simulated dataset.
    """
    mean_a = [0, 0]
    mean_b = [4, 7]
    sigma = np.eye(2)
    samples_a = np.random.multivariate_normal(mean_a, sigma, 150)
    samples_b = np.random.multivariate_normal(mean_b, sigma, 150)
    X_tr = np.concatenate((samples_a[0:50, :], samples_b[0:50, :]), axis=0)
    X_val = np.concatenate((samples_a[50:100, :], samples_b[50:100, :]), axis=0)
    X_te = np.concatenate((samples_a[100:150, :], samples_b[100:150, :]), axis=0)
    Y_tr = np.concatenate((np.zeros(50), np.ones(50)))
    Y_tr = to_categorical(Y_tr, num_classes=2)
    Y_val = copy.deepcopy(Y_tr)
    Y_te = copy.deepcopy(Y_tr)
    X = np.concatenate((X_tr, X_val, X_te), axis=0)
    adj = squareform(pdist(X, 'Euclidean'))

    return adj, X, X_tr, Y_tr, X_val, Y_val, X_te, Y_te


def preprocess_sim(adj, features):
    """
    Pre-process the simulated dataset
    """
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    A = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    # Row-normalize training samples
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return A, features


class ModelException(Exception):
    def __init__(self, msg='Model construction exception.'):
        self.message = msg


class LayerException(Exception):
    def __init__(self, msg="Layer construction exception."):
        self.message = msg


class Mymodel(tf.keras.Model):
    def __init__(self, networks, adj, dropout_rate=0.5, weight_decay=5e-4):
        super(Mymodel, self).__init__(name='Mymodel')
        # Networks should be a list
        if not isinstance(networks, list):
            raise (ModelException("The argument networks should be a list"))
        self.networks = networks
        self.adj = adj
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        # Construct layers
        self._layers = []
        for _net in networks[:-1]:
            layer = tf.keras.layers.Dropout(self.dropout_rate)
            self._layers.append(layer)
            layer = Mydense(_net, adj=self.adj)
            self._layers.append(layer)

        # The last layer
        self._layers.append(tf.keras.layers.Dropout(self.dropout_rate))
        layer = Mydense(networks[-1], adj=self.adj,
                        activation=lambda x: x)
        self._layers.append(layer)

    def call(self, inputs, training=False, **kwargs):
        if training:
            x = self._layers[0](inputs)
            for _layer in self._layers[1:]:
                x = _layer(x)
        else:
            # The 1st, 3rd, 5th,... etc are dropout layers
            x = self._layers[1](inputs)
            for _layer in self._layers[3::2]:
                x = _layer(x)
        return tf.keras.activations.softmax(x, axis=-1)


class Mydense(tf.keras.layers.Layer):
    def __init__(self, output_dim, adj, activation=tf.keras.activations.relu,
                 **kwargs):
        self.activation = activation
        self.output_dim = output_dim
        self.adj = adj
        super(Mydense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='w',
                                 shape=(input_shape[1], self.output_dim),
                                 initializer=tf.keras.initializers.glorot_uniform(),
                                 trainable=True)
        super(Mydense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        return self.activation(tf.matmul(self.adj.astype('float32'), tf.matmul(x, self.w)))

# L2 regularized loss
def gcnloss(model, ground_truth, pred):
    # Only the first layer is regularised
    loss = model.weight_decay * tf.nn.l2_loss(model._layers[1].w)

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    loss += loss_object(ground_truth, pred)
    return loss


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("cora")
# adj, X,  X_tr, Y_tr, X_val, Y_val, X_te, Y_te = gen_data()
# Pre-process the training dataset
A, X = preprocess(adj, features)

# Train a model
epochs = 200
old_loss_val = 1000.
layer1_units = 16
layer2_units = np.shape(y_train)[-1]
dropout_rate = 0.5

tf.keras.backend.set_floatx('float32')
model = Mymodel([layer1_units, layer2_units], A, dropout_rate=dropout_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

count = 0
print("Iteration begin:")
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        pred = model(X, training=True)
        loss = gcnloss(model, y_train[train_mask, :], pred[train_mask])
    # Gradient
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update weights within our model
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Early stopping
    pred = model(X, training=False)
    loss_val = gcnloss(model, y_val[val_mask, :], pred[val_mask])
    if loss_val > old_loss_val:
        count += 1
        print("------------------------------------------------------")
        print("Loss ascending count: %d, epoch: %d, loss: %.4f" % (count, epoch, loss_val))
        if count >= 10:
            break
    else:
        count = 0
        if epoch % 5 == 0:
            print("Epoch: %d, loss: %.4f" % (epoch, loss_val))
    old_loss_val = loss_val

# Estimation on the test dataset
print("------------------------------------------------------")
pred = model(X, training=False)
loss_test = gcnloss(model, y_test[test_mask, :], pred[test_mask])
print("Loss on test dataset: %.4f" % loss_test)
accu = categorical_accuracy(y_test[test_mask, :], pred[test_mask])
print("Accuracy: %.4f" % (np.sum(accu) / len(accu)))
