import numpy as np
from random import shuffle
# from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)  # (1,D)*(D,C)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:, y[i]] -= X[i]
        dW[:, j] += X[i]
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW /= num_train
  dW += 2 * W * reg

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  scores = X.dot(W) # N, C
  num_class = W.shape[1]
  num_train = X.shape[0]
  correct_class_score = scores[np.arange(num_train), y] # 使用数组索引,下标为(i,y_i) , shape: (num_class,)   
  correct_class_score = correct_class_score.reshape([-1, 1]) # shape: (num_class,1) 
  loss_tep = scores - correct_class_score + 1
  loss_tep[loss_tep < 0] = 0  # max(0,loss_tep)
  # loss_tep = np.maximum(0, loss_tep)
  loss_tep[np.arange(num_train), y] = 0 # 正确的类loss为0
  loss = loss_tep.sum()/num_train + reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass

  # loss_tep等于0的位置，对X的导数就是0
  # loss_tep元素大于0的位置，说明此处有loss，对于非正确标签类的S求导为1
  loss_tep[loss_tep > 0] = 1  # N, C  
  # 对于正确标签类，每有一个loss_tep元素大于0，则正确标签类的S求导为-1，要累加
  loss_item_count = np.sum(loss_tep, axis=1)     
  loss_tep[np.arange(num_train), y] -= loss_item_count  #在一次错误的分类中，
  # dW中第i,j元素对应于第i维，第j类的权重
  # X.T的第i行每个元素对应于每个样本第i维的输入，正是Sj对W[i,j]的导数
  # loss_tep的第j列每个元素对应于每个样本在第j类的得分是否出现，相当于掩码
  # X.T和loss_tep的矩阵乘法等于对每个样本的W[i,j]导数值求和
  # 简单地说，就是loss对S的导数是loss_tep， loss_tep对W的导数是X.T
  dW = X.T.dot(loss_tep) / num_train  # (D, N) *(N, C) 
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
