import numpy as np
import matplotlib.pyplot as plt

def huber_loss(y, delta, t=0):
    a = y - t
    if np.abs(a) <= delta:
        loss = 0.5 * a ** 2
    else:
        loss = delta * (np.abs(a) - 0.5 * delta)
    return loss

def se_loss(y, t=0):
    loss = 0.5 * (y - t) ** 2
    return loss

def main():
    ys = np.linspace(-400, 400, 100)
    ms = list(map(se_loss, ys))
    plt.plot(ys, ms, label="Squared Error")
    
    for delta in [1, 10, 50, 100]:
        hs = list(map(lambda y: huber_loss(y, delta=delta), ys))
        plt.plot(ys, hs, label=f"Huber, delta={delta}")
    
    plt.legend(loc="upper right")
    plt.xlabel('y')
    plt.ylabel('Loss')

    plt.show()

if __name__ == "__main__":
    main()