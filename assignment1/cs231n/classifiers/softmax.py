import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  # gradient trick: see http://cs231n.github.io/neural-networks-case-study/
  for i in range(num_train):
    scores = np.dot(X[i], W)
    # !!! The trick for numerical stability, see http://cs231n.github.io/linear-classify
    scores -= np.max(scores)   
    sum_j = np.sum(np.exp(scores))
    p = lambda k: np.exp(scores[k]) / sum_j
    loss += -np.log(p(y[i]))
    
    for k in range(num_classes):
      dW[:, k] += (p(k) - (y[i] == k)) * X[i]
  
  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]  

  scores = np.dot(X, W)
  scores -= np.max(scores, axis=1)[:, np.newaxis]
  sum_j = np.sum(np.exp(scores), axis=1)[:, np.newaxis]
  p = np.exp(scores) / sum_j
  loss = np.sum(-np.log(p[np.arange(num_train), y]))
    
  indicator = np.zeros_like(p)
  indicator[np.arange(num_train), y] = 1
  dW = np.dot(X.T, (p - indicator))
  
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

