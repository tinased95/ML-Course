import numpy as np

def gradient_descent(X, t, B, W, num_iter=10, learning_rate=0.001, delta=1):
    y = np.zeros(len(t)) # prediction
    for i in range (num_iter):
        y = np.dot(X, W) + B
        print("y: ", y)
        a = y - t
        W = W - learning_rate * dL_dW(a, X, delta)
        B = B - learning_rate * dL_dB(a, X, delta)

    return W, B

def dL_dW(a, X, delta):
    cost = np.where(np.abs(a) < delta, a, delta * a / np.abs(a))
    cost = np.dot(cost.T, X).T
    # print("W: " , cost)
    return cost

def dL_dB(a, X, delta):
    cost = np.where(np.abs(a) < delta, a, delta * a / np.abs(a))
    # print("B: ", cost)
    return cost

def main():
    N = 4  # number of samples
    D = 2 # number of features

    # initializing w and b
    B = np.zeros((N,1))
    W = np.zeros((D,1))

    W_t = np.random.randn(D,1)
    B_t = np.random.randn(N,1)


    # initializing X
    X = np.random.randn(N, D)

    # initializing target y = 4x + 20
    # epsilon is randomly sampled noise from a standard normal N(0,1)
    #epsilon = 0  
    t = np.dot(X, W_t) + B_t 

    weight, bias = gradient_descent(X, t, B, W)
    print("optimal weight: " , weight)
    print("optimal bias: ", bias)

if __name__ == "__main__":
    main()

