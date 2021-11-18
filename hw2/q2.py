from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.misc import logsumexp


#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist


def predict(x_test, x_train, db_weight, y_train, y_test, lam):
    test_num, feature_num = x_test.shape
    h = np.zeros([test_num, 1])
    for t in range(test_num):
        A = np.diag(db_weight[t,:])
        A1 = (((x_train.T).dot(A)).dot(x_train)) + lam * np.identity(feature_num)
        B = ((x_train.T).dot(A)).dot(y_train)
        W = np.linalg.solve(A1, B)
        h[t,0] = ((x_test[t,:]).T).dot(W)

    loss = ((h - y_test.reshape(-1,1))**2).mean()
    return h, loss


# perform LRLS for each of the test data in x_test (N_test x d)
def LRLS(x_test, x_train, y_test, y_train, taus):
    loss = np.zeros(taus.shape)
    for i, tau in enumerate(taus):
        dist = l2(x_test, x_train)
        dist_tau = -dist / (2 * tau**2)
        dist_tau -= (np.max(dist_tau, axis=1)).reshape(-1,1)
        dist_exp = np.exp(dist_tau)
        db_weight = (dist_exp) / np.sum(dist_exp, axis=1).reshape(-1,1)
        h, loss[i] = predict(x_test, x_train, db_weight, y_train.reshape(-1,1), y_test, lam=0.003)
    return loss


def run_k_fold(x, y, taus, fold_num):
    fold_length = int(x.shape[0] / (float(fold_num)))
    loss = np.zeros((fold_num, taus.shape[0]))
    for k in range(fold_num):
        # pick the validation set
        x_validation = x[k*fold_length:(k+1)*fold_length, :]
        y_validation = y[k*fold_length:(k+1)*fold_length, :]
        # pick the training set
        x_train = np.delete(x, range(k*fold_length, (k+1)*fold_length), axis=0)
        y_train = np.delete(y, range(k*fold_length, (k+1)*fold_length), axis=0)
        #
        loss[k,:] = LRLS(x_validation, x_train, y_validation, y_train, taus)

    loss_mean = loss.mean(axis=0)
    return loss_mean


def main():

    np.random.seed(0)
    # load boston housing dataset
    boston = load_boston()
    x = boston['data']
    N = x.shape[0]

    # feature standardization
    # x = x - x.mean(axis=1).reshape(-1,1)
    # x = x / x.std(axis=1).reshape(-1,1)

    # Add a column of ones to X (adding bias to our model)
    x = np.concatenate((np.ones((506,1)),x),axis=1)
    d = x.shape[1]
    y = boston['target']
    feature_num = x.shape[1]

    # split data into training and test set
    idx = np.random.permutation(range(N))
    # Use %80 of dataset for training
    train_num = int(np.round(0.8 * x.shape[0]))
    # Pick training data
    x_train = x[idx[0:train_num],:]
    y_train = y[idx[0:train_num]]
    # Pick test data
    x_test = x[idx[train_num:x.shape[0]],:]
    y_test = y[idx[train_num:x.shape[0]]]

    taus = np.logspace(1,3,50)
    # run k-fold cross validation on the training dataset (randomly chosen %80)
    loss = run_k_fold(x_train, y_train.reshape(-1,1), taus, 5)

    # find the best tau
    tau_opt = taus[np.argmin(loss)].reshape(1,1)

    # test the model (with best tau)
    loss_min = LRLS(x_test, x_train, y_test, y_train, tau_opt)

    # visualize loss value with respect to tau
    plt.plot(taus, loss, lw=2)
    plt.grid(color='black', lw=2)
    plt.xticks(np.arange(0, 1000, 80))
    plt.xlabel('tau')
    plt.ylabel('Loss(MSE)')
    plt.show()

if __name__ == "__main__":
    main()
