import time
import numpy as np
import matplotlib.pyplot as plt


def main():
    # prepare data in python list
    mylist = []
    with open('ex1data1.txt') as f:
        for line in f.readlines():
            tmp = line.strip().split(',')
            mylist.append([float(tmp[0]), float(tmp[1])])

    # create numpy array
    data = np.array(mylist)

    # prepare X and y
    x = data[:, 0]
    y = data[:, 1]
    m = len(y)
    X = np.concatenate([np.ones(shape=(m, 1)), np.reshape(data[:, 0], (m, 1))], axis=1)

    # initialize theta
    theta = np.zeros((2, 1))

    # set learning rate and numer of iterations
    iterations = 1500
    alpha = 0.01

    start_time = time.time()
    for step in range(iterations):
        cost = 0
        val = 0
        for i in range(m):
            val = np.dot(X[i, :], theta)
            cost += (val-y[i])**2
        print("cost is ", cost/(2*m))
        val = 0
        for i in range(m):
            val += (np.dot(X[i, :], theta) - y[i])*X[i, 0]
        theta[0] = theta[0] - alpha/m*val

        for i in range(m):
            val += (np.dot(X[i, :], theta) - y[i])*X[i, 1]
        theta[1] = theta[1] - alpha/m*val

    print("--- {} seconds ---".format(time.time() - start_time))
    print(theta)

    # plot data and result
    res_x = np.linspace(0, 25, 100)
    res_y = theta[0]+res_x*theta[1]
    plt.plot(x, y, 'x')
    plt.plot(res_x, res_y)
    plt.show()

if __name__ == "__main__":
    main()
