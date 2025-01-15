
#######################################################################
# Learning MNIST dataset with 1 hidden layer fully connected network  #
# "by hand", that is using only Google JAX's numpy-equivalent for GPU #
# acceleration. Forward propagation and backpropagation both          #  
# explicitly calculated in terms of the appropriate matrices.         #
#                                                                     #
# Achieving accuracy >95%                                             #
#                                                                     #
# CC-BY-NC-4.0                                                         #
#######################################################################

import jax
from jax import random
key = jax.random.PRNGKey(42)
import jax.numpy as jn
#%matplotlib inline

def sigmoid(x):
    return 1.0/(1.0 + jn.exp(-x))

def one_hot(y):
    return jax.nn.one_hot(y,10)

def cross_entropy_loss(logits, targets):
    lsml = jax.nn.log_softmax(logits)
    return -jn.mean(jn.sum(targets*lsml,axis=1))

# Load the MNIST dataset
import tensorflow.keras as keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

# DEFINE THE MODEL (=two matrices with bias)
var = 0.01
input_dim = 28*28
hidden_layer_dim = 100
output_dim = 10
# initialize input layer:
W0 = random.normal(key, shape=(input_dim, hidden_layer_dim)) * jn.sqrt(var)
new_row = random.normal(key, shape=(1, hidden_layer_dim)) * jn.sqrt(var)
# extended matrix to incorporate the bias unit
W0e = jn.vstack([W0, new_row])
# initialize output layer
W1 = random.normal(key, shape=(hidden_layer_dim,output_dim)) * jn.sqrt(var)
new_row = random.normal(key, shape=(1, output_dim)) * jn.sqrt(var)
# extended matrix to incorporate the bias unit
W1e = jn.vstack([W1, new_row])

# forward pass:
def forward(I,T):
    bs = I.shape[0]
    Ie = jn.hstack([I,jn.ones((bs, 1))])
    o1=sigmoid(Ie@W0e)
    Do1=o1*(1-o1)
    o1e = jn.hstack([o1,jn.ones((bs,1))])
    o2 = o1e@W1e
    loss = cross_entropy_loss(o2,T)
    return Do1, o1e, o2, loss

def backprop(I,Do1,o1e,o2,T):
    bs = I.shape[0]
    Ie = jn.hstack([I,jn.ones((batch_size, 1))])
    Dloss = jax.grad(cross_entropy_loss, argnums=0)
    Dl=Dloss(o2,T)
    o1e=o1e.reshape((bs,101,1))
    Dl=Dl.reshape((bs,1,10))
    Grad_wrt_W1e = (o1e@Dl).mean(axis=0)
    Dl=Dl.reshape((bs,10,1))
    D1_=jn.einsum('ij,kjl->kil',W1,Dl)
    D1=Do1*D1_.reshape((bs,100))
    Ie=Ie.reshape(bs,785,1)
    D1=D1.reshape(bs,1,100)
    Grad_wrt_W0e = (Ie@D1).mean(axis=0)
    return Grad_wrt_W0e, Grad_wrt_W1e

# Learning rate
alpha = 500.0
for k in range(200):
    perm = jax.random.permutation(key, x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]
    batch_size = 10_000
    # smaller batches make the gradient descent faster in towards the end:
    if k > 40: batch_size = 6000
    if k > 80: batch_size = 3000
    if k > 100: batch_size = 1000
    n_batches = x_train.shape[0] // batch_size
    x_batches = x_train[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    y_batches = y_train[:n_batches * batch_size].reshape(n_batches, batch_size)
    # decrease learning rate towards the end:
    alpha = 100_000.0
    if loss < 1.0: alpha = 10_000
    if loss < 0.5: alpha = 5_000
    if loss < 0.2: alpha = 1_000
    if loss < 0.1: alpha = 500
    if loss < 0.05: alpha = 100
    if loss < 0.03: alpha = 10
    for i in range(x_batches.shape[0]):
        I = x_batches[i]
        T = one_hot(y_batches[i])
        Do1, o1e, o2, loss = forward(I,T)
        DW0e, DW1e = backprop(I,Do1,o1e,o2,T)
        # Update "momentum":
        try:
            MW0e = 0.8*MW0e + 0.2*DW0e
            MW1e = 0.8*MW1e + 0.2*DW1e
        except:
            MW0e = DW0e.copy()
            MW1e = DW1e.copy()
        # Update weights with momentum:
        W0e -= alpha*MW0e
        W1e -= alpha*MW1e
        W0 = W0e[:-1]
        W1 = W1e[:-1]
        if i==0:
            _,_,o2,loss=forward(x_train.reshape(x_train.shape[0],-1),one_hot(y_train))
            print("epoch",k,"loss:",loss,"training acc:",jn.mean(o2.argmax(axis=1)==y_train))

# Test:
I_test = x_test.reshape(x_test.shape[0],-1)
T_test = one_hot(y_test)
_,_,o2,loss = forward(I_test,T_test)
print("loss:",loss,"test acc:",jn.mean(o2.argmax(axis=1)==y_test))

