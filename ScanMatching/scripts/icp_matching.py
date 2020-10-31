# Import numpy package
import numpy as np
# Import library to plot in python
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from matplotlib.pyplot import draw
# Import functions from scikit-learn
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error
import random
import cv2


def UpdateTransformationMatrices(R_prev, T_prev, R, T):
    '''
    Computes the least-squares best-fit transform that maps corresponding points P to X.
    Inputs :
        R_prev = (2 x 2) matrix representing the rotation for the previous iteration
        T_prev = (2 x 1)matrix representing the translation for the previous iteration
    Returns :
        R = (2 x 2) matrix representing the global rotation
        T = (2 x 1) matrix representing the global translation
        Such that R * X + T is aligned on P
    '''
    # if(T_prev is None):
    #     T_global = T
    #     R_global = R
    #     return R_global, T_global

    H_prev = [[R_prev[0,0], R_prev[0,1], T_prev[0,0]],
              [R_prev[1,0], R_prev[1,1], T_prev[1,0]],
              [0, 0, 1]]
    H = [[R[0, 0], R[0, 1], T[0, 0]],
         [R[1, 0], R[1, 1], T[1, 0]],
         [0, 0, 1]]

    H_global = np.dot(H_prev, H)

    R_global = np.array([[H_global[0, 0], H_global[0, 1]],
                        [H_global[1, 0], H_global[1, 1]]])
    T_global = np.array([[H_global[0, 2]],
                        [H_global[1, 2]]])

    return R_global, T_global


def ComputeBestTransformation(P, X):
    '''
    Computes translation and rotation that minimizes the sum of the squared error between points X and P
    Inputs :
        X = input points
        P = reference scene points
    Returns :
        R = (d x d) rotation matrix
        T = (d x 1) translation vector
        Such that R * X + T is aligned on P
    '''

    # Compute the center of mass of each set of points
    u_p = np.mean(P, axis=1)
    u_x = np.mean(X, axis=1)

    # Substract center of mass in the two sets
    P_prime = (P.T - u_p).T
    X_prime = (X.T - u_x).T

    # Get matrix W = Q'xQ^t
    W = np.dot(X_prime, P_prime.T)

    # Find the Singular-Value-Descomposition --> ( U*S*V^t ) of H
    U, S, V_t = np.linalg.svd(W)

    # Return R = UV^t and T = u_x - R*u_p'
    R = np.dot(U, V_t)

    # special reflection case
    if np.linalg.det(R) < 0:
        U[:,len(U)-1] = U[:,len(U)-1]*(-1)
        R = np.dot(U, V_t)

    T = u_x - (np.dot(R, u_p))

    return R, np.expand_dims(T, axis=1)


def ICPMatching(P, X, max_iter, RMS_threshold, fast = False, n_samples = 200):
    '''
    ICP algorithm with a point to point strategy applying Nearest neghbor search.
    Inputs :
        X = input points
        P = reference scene points
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        points_aligned = X points aligned on P points

    '''
    # To compute the final transformation at the end
    X_ori = X

    # Initialize variables
    R_glob = np.eye(2)
    T_glob = np.zeros((2, 1))

    # Crating the KDTree from reference pointcloud
    P_KDTree = KDTree(P, leaf_size=10)

    # Ask for the nearest neighbor of each point in X
    _, idx = P_KDTree.query(X, k=1)

    # Create plot objects
    fig, ax = plt.subplots()
    for iteration in range(max_iter):

        if (fast):
            X_sample = X[ np.random.choice(len(X), n_samples, replace=False),:]
        else:
            X_sample = X

        # Query for 1-nearest neighbors of each point
        _, idx = P_KDTree.query(X_sample, k=1)

        # Applying best_rigid_transform in order to find R and T
        R, T = ComputeBestTransformation(X_sample.T, np.vstack(P[idx]).T)

        # Applying the transformation
        X = (np.dot(R, X.T) + T).T

        # Accumulate the transformations
        # Does not work propertly, intead we re-compute at the end the global transformation
        R_glob, T_glob = UpdateTransformationMatrices(R_glob, T_glob, R, T)

        # Computing the RMS error
        rms_error =  mean_squared_error(np.vstack(P[idx]), X_sample)
        print('iteration [{:d}] - The RMS (root mean square) error is: ({:.4f})'.format(iteration, rms_error))

        if rms_error < RMS_threshold:
            break

        # Plot
        plt.cla()
        ax.plot(X.T[0], X.T[1], 'r.')
        ax.plot(P.T[0], P.T[1], 'b.')
        ax.set_title('ICP matching 2D')
        ax.set_title('Iteration {:d}  RMS error {:0.4f}'.format(iteration, rms_error))
        plt.axis('equal')
        plt.draw()
        plt.waitforbuttonpress(0) # this will wait for indefinite time

    plt.show()

    # Variable for aligned X, this is the cloud after applying ICP
    X_aligned = np.copy(X)

    # problem with Transformation Accumulation, instead we recalculate the transformation
    # between the original input points to the final transformed points
    R, T = ComputeBestTransformation(X_ori.T, X.T)

    return X_aligned, R, T

def create_points(n_points):
    #for i in range(n_points):
    data = np.random.rand(n_points, 2)*100
    rad = np.random.random_sample()*40
    tx = np.random.random_sample()*50
    ty = np.random.random_sample()*50

    theta = np.radians(rad)
    c, s = np.cos(theta), np.sin(theta)

    H = [[c, -s, tx],
         [s, c, ty],
         [0, 0, 2]]

    R = np.array([ [H[0][0], H[0][1]], [H[1][0], H[1][1]] ])
    T = np.array([ [H[0][2]], [H[1][2]] ])

    template = (np.dot(R, data.T).T + T.T)

    return data, template


if __name__ == '__main__':

    points_ref2, points2 = create_points(1000)

    # Create and save data
    # np.save('points_ref.npy', points_ref2)
    # np.save('points.npy', points2)

    # # Load data
    # points_ref = np.load('points_ref.npy')
    # points = np.load('points.npy')
    random.shuffle(points2)

    # Apply ICP
    max_iter = 100
    RMS_threshold = 0.01
    n_samples = 200
    data_aligned, R, T = ICPMatching(points_ref2, points2, max_iter, RMS_threshold)

    # fig, ax = plt.subplots()
    # plt.cla()
    # ax.plot(points.T[0], points.T[1], 'r.')
    # ax.plot(points_ref.T[0], points_ref.T[1], 'b.')
    # ax.set_title('ICP matching 2D')
    # plt.axis('equal')
    # plt.draw()
    # plt.show()
