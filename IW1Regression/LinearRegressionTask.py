import numpy as np

def normalize(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0) 
    X_new = (X-mean)/std 
    
    return X_new, mean, std

def prepare_X(X):
    m = X.shape[0]
    ones = np.ones((m, 1))
    X_new = np.array(X[:])
    X_new = np.column_stack((ones, X_new))
    return X_new

def hypothesis(X, theta):
    h_thetha = np.dot(theta, X.transpose())
    return h_thetha


def cost_function(X, y, theta):
    m = X.shape[0]
    if m == 0:
        return None
    J = (1/(2*m)) * np.sum(np.square((hypothesis(X,theta) - y)))
    return J



def derivative(X, y, theta):
    m = X.shape[0]

    d_thetha = (1/m) * ((hypothesis(X, theta) - y) @ X)

    return d_thetha


def gradient_descent(X, y, theta, alpha, num_iters, print_J = True):
    m = X.shape[0]
    J_history = []
    J = cost_function(X, y, theta)
    if print_J == True:
        print(J)
    J_history.append(J)

    for i in range(num_iters):
        delta = derivative(X,y,theta)
        theta = theta - alpha*delta
        J = cost_function(X, y, theta)

        if (J - J_history[-1] > 1e-6):
            break

        if print_J == True:
            print(J)
        J_history.append(J)
        
    return theta, J_history