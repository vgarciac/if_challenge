# Import numpy package
import numpy as np
# Import library to plot in python
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from matplotlib.pyplot import draw
# Import functions from scikit-learn
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error
# Import function for random generation
import random
import pandas as pd

def UpdateTransformationMatrices(R_global, T_global, R, T):
    '''
    Computes the best-fit transform that maps corresponding points P to X.
    Inputs :
        R_global = (2 x 2) matrix representing the global rotation
        T_global = (2 x 1) matrix representing the global translation
        R = (2 x 2) matrix representing the rotation for the current iteration
        T = (2 x 1) matrix representing the translation for the current iteration
    Returns :
        R = (2 x 2) matrix representing the global rotation
        T = (2 x 1) matrix representing the global translation
        Such that R * X + T is aligned on P
    '''
    # if(T_global is None):
    #     T_global = T
    #     R_global = R
    #     return R_global, T_global

    # Build global transformation --> H = [R T]
    H_global = [[R_global[0,0], R_global[0,1], T_global[0,0]],
              [R_global[1,0], R_global[1,1], T_global[1,0]],
              [0, 0, 1]]

    # Build current iteration transformation --> H = [R T]
    H = [[R[0, 0], R[0, 1], T[0, 0]],
         [R[1, 0], R[1, 1], T[1, 0]],
         [0, 0, 1]]

     # update global transformation --> H_glob = H_glob * H
    H_global = np.dot(H_global, H)

    # Separate R from H
    R_global = np.array([[H_global[0, 0], H_global[0, 1]],
                        [H_global[1, 0], H_global[1, 1]]])
    # Separate T from H
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
        R = (2 x 2) rotation matrix
        T = (2 x 1) translation vector
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

    # Return R = UV^t
    R = np.dot(U, V_t)

    # special reflection case
    if np.linalg.det(R) < 0:
        U[:,len(U)-1] = U[:,len(U)-1]*(-1)
        R = np.dot(U, V_t)

    # Return T = u_x - R*u_p'
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
        fast = boolean to activate the subsampling option (faster for big sets)
    Returns :
        points_aligned = X points aligned to P
    '''
    # X_original -> To compute the final transformation at the end
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
        ax.plot(P.T[0], P.T[1], 'b,')
        ax.plot(X.T[0], X.T[1], 'r,')
        ax.set_title('ICP matching 2D')
        ax.set_title('Iteration {:d}  RMS error {:0.6f}'.format(iteration, rms_error))
        plt.axis('equal')
        plt.draw()
        # Wait for button to avance one itaration
        plt.waitforbuttonpress(0)

    plt.show()

    # Variable for aligned X, this is the cloud after applying ICP
    X_aligned = np.copy(X)

    # problem with Transformation Accumulation, instead we recalculate the transformation
    # between the original input points to the final transformed points
    R, T = ComputeBestTransformation(X_ori.T, X.T)

    return X_aligned, R, T

def transformPoints(points):

    rad = np.random.random_sample()*40
    tx = np.random.random_sample()*50
    ty = np.random.random_sample()*50

    theta = np.radians(rad)
    c, s = np.cos(theta), np.sin(theta)

    H = [[c, -s, tx],
         [s, c, ty],
         [0, 0, 2]]

    R = np.array([ [H[0][0], H[0][1]],
                   [H[1][0], H[1][1]] ])
    T = np.array([ [H[0][2]], [H[1][2]] ])

    points_transformed = (np.dot(R, points.T).T + T.T)

    return points_transformed

def create_points(n_points):
    #for i in range(n_points):
    ref_pts = np.random.rand(n_points, 2)*100
    
    pts = transformPoints(ref_pts)

    return ref_pts, pts


if __name__ == '__main__':
    
    # Create data
    points_ref, points = create_points(1000)
    # random.shuffle(points)

    # Save data
    #######################################################################
    # np.save('points_ref.npy', points_ref)
    # np.save('points.npy', points)

    # Load data
    #######################################################################
    # points_ref = np.load('points_ref.npy')
    # points = np.load('points.npy')

    # Load real data
    #######################################################################
    # df = pd.read_csv (r'../pointclouds/Hokuyo_0.csv')
    # # Cut the set of points
    # points_ref = df[df['x'] >= 0.5]
    # points_ref = points_ref.iloc[:, [1,2]]
    # points_ref = points_ref.to_numpy()
    # # Apply transformation to the points
    # points = transformPoints(points_ref)
    # # multiply points for debug purposes 
    # points = points*5
    # points_ref = points_ref*5

    # Run the algorithm
    #######################################################################
    # Define parameters
    max_iter = 1000
    RMS_threshold = 0.0001
    n_samples = 500
    # Apply ICP
    data_aligned, R, T = ICPMatching(points_ref, points, max_iter, RMS_threshold)