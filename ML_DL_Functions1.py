import numpy as np
def ID1():
    '''
        Write your personal ID here.
    '''
    # Insert your ID here
    return 315875385
def ID2():
    '''
        Only If you were allowed to work in a pair will you fill this section and place the personal id of your partner otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

def LeastSquares(X,y):
  '''
    Calculates the Least squares solution to the problem
    X*theta=y using the least squares method
    :param X: input matrix
    :param y: input vector
    :return: theta = (Xt*X)^(-1) * Xt * y 
  '''
  nt = X.shape[0]
  X = np.hstack([X,np.ones((nt,1))])
  theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),y)
  w = theta[:-1]
  b = theta[-1]
  return [w,b]

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: input matrix
    :param s: input ground truth label
    :return: accuracy of the model
  '''
  pred_s = model.predict(X)
  total_pred = len(s)
  correct_pred = np.sum(pred_s == s)
  return 100*correct_pred/total_pred

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem.  
  '''
  return [-6.84638411e-02,  1.69484479e-02, -3.80077855e-02,  7.69957682e-03,
  1.56273883e-03, -1.91986315e-02,  8.22773626e-02,  2.80468533e-02,
  2.40642502e-03, -3.86323535e-02,  5.56648929e-02,  6.92378333e-03,
  9.04490111e-02,  1.75280352e-01,  7.48338913e-01,  3.84442304e-02,
 -1.80370777e-03, -4.09779798e-03,  3.30490125e-02, -8.34425392e-04,
  2.68985072e-02, -4.94240547e-03,  6.02786379e-04, -1.80211765e-02,
 -2.16350409e-02,  7.45105292e-03, -2.81659958e-02, -2.01177337e-02]

def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return 1.383781094370375e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the classification problem.  
  '''
  return [[-0.32030079, -0.42772193,  0.45104687,  0.05220711, -0.27712585, -0.42367126,
   0.0302132,   0.10011699,  0.12515704,  0.28244838, -0.65204097, -0.21885506,
   0.05817121,  1.17389897,  2.90074388, -0.59009519, -0.10350768, -0.09441491,
  -0.12099739,  0.0284501,  -0.11743172,  0.19156617, -0.19678774, -0.25771801,
  -0.2457699,   0.26426237,  0.08429817,  0.46676014]]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return [0.48464441]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem.  
  '''
  return [0, 1]