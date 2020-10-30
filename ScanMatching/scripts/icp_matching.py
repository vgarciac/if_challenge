# Import numpy package
import numpy as np
# Import library to plot in python
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from matplotlib.pyplot import draw
# Import functions from scikit-learn (Kdtree)
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error

def show_ICP(data, ref, R_list, T_list, neighbors_list):
    '''
    Show a succession of transformation obtained by ICP.
    Inputs :
                  data = (d x N_data) matrix where "N_data" is the number of point and "d" the dimension
                   ref = (d x N_ref) matrix where "N_ref" is the number of point and "d" the dimension
                R_list = list of the (d x d) rotation matrices found at each iteration
                T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = list of the neighbors of data in ref at each iteration

    This function works if R_i and T_i represent the transformation of the original cloud at iteration i, such
    that data_(i) = R_i * data + T_i.
    If you saved incremental transformations such that data_(i) = R_i * data_(i-1) + T_i, you will need to
    modify your R_list and T_list in your ICP implementation.
    '''

    # Get the number of iteration
    max_iter = len(R_list)

    # Get data dimension
    dim = data.shape[0]

    # Insert identity as first transformation
    R_list.insert(0, np.eye(dim))
    T_list.insert(0, np.zeros((dim, 1)))

    # Create global variable for the graph plot
    global iteration, show_neighbors
    iteration = 0
    show_neighbors = 0

    # Define the function drawing the points
    def draw_event():
        data_aligned = R_list[iteration].dot(data) + T_list[iteration]
        plt.cla()
        if dim == 2:
            ax.plot(ref[0], ref[1], '.')
            ax.plot(data_aligned[0], data_aligned[1], '.')
            if show_neighbors and iteration < max_iter:
                lines = [[data_aligned[:, ind1], ref[:, ind2]] for ind1, ind2 in enumerate(neighbors_list[iteration])]
                lc = mc.LineCollection(lines, colors=[0, 1, 0, 0.5], linewidths=1)
                ax.add_collection(lc)
            plt.axis('equal')
        if dim == 3:
            ax.plot(ref[0], ref[1], ref[2], '.')
            ax.plot(data_aligned[0], data_aligned[1], data_aligned[2], '.')
            plt.axis('equal')
        if show_neighbors and iteration < max_iter:
            ax.set_title('Iteration {:d} ===> press right / left to change\nNeighbors ON ===> Press n to change (only in 2D)'.format(iteration))
        else:
            ax.set_title('Iteration {:d} ===> press right / left to change\nNeighbors OFF ===> Press n to change (only in 2D)'.format(iteration))

        plt.draw()

    # Define the function getting keyborad inputs
    def press(event):
        global iteration, show_neighbors
        if event.key == 'right':
            if iteration < max_iter:
                iteration += 1
        if event.key == 'left':
            if iteration > 0:
                iteration -= 1
        if event.key == 'n':
            if dim < 3:
                show_neighbors = 1 - show_neighbors
        draw_event()

    # Create figure
    fig = plt.figure()

    # Intitiate graph for 3D data
    if dim == 2:
        ax = fig.add_subplot(111)
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        print('wrong data dimension')

    # Connect keyboard function to the figure
    fig.canvas.mpl_connect('key_press_event', press)

    # Plot first iteration
    draw_event()

    # Start figure
    plt.show()


def draw_event1():
    plt.show()

def press1(event):
    if event.key == 'right':
        pass
    draw_event1()
    
def calculate_global_matrices(R_prev, T_prev, R, T):


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


def best_rigid_transform(P, X):
    '''
    Computes the least-squares best-fit transform that maps corresponding points P to X.
    X = reference scene points
    P = input points
    Inputs :
        X = (d x N) matrix where "N" is the number of points and "d" the dimension
        P = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
        R = (d x d) rotation matrix
        T = (d x 1) translation vector
        Such that R * data + T is aligned on ref
    '''

    # YOUR CODE


    # 1) Calculate barycenter u_w and u_w
    u_p = np.mean(P, axis=1)
    u_x = np.mean(X, axis=1)

    # 2) Compute centered clouds Q and Q'
    P_prime = (P.T - u_p).T
    X_prime = (X.T - u_x).T

    # 3) Get matrix H = Q'xQ^t
    W = np.dot(X_prime, P_prime.T)

    # 4) Find the SINGULAR-VALUE DECOMPOSITION --> ( U*S*V^t ) of H
    U, S, V_transposed = np.linalg.svd(W)

    # 5) Return R = VU^t and T = pm - Rpm'
    R = np.dot(U, V_transposed)
    if np.linalg.det(R) < 0:
        U[:,len(U)-1] = U[:,len(U)-1]*(-1)
        R = np.dot(U, V_transposed)
    T = u_x - (np.dot(R, u_p))

    return R, np.expand_dims(T, axis=1)

def icp_matching(X, ref, max_iter, RMS_threshold):
    '''
    ICP algorithm with a point to point strategy applying Nearest neghbor search.
    Inputs :
        X = (d x N_X) matrix where "N_X" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        points_aligned = X points aligned on P points
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each X point in
        the ref cloud and this obtain a (1 x N_X) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Initialize variables
    R_prev = np.eye(2)
    T_prev = np.zeros((2, 1))

    # Crating the KDTree from reference pointcloud
    ref_KDTree = KDTree(ref, leaf_size=10)

    # Ask for the nearest neighbor of each point in X
    _, index = ref_KDTree.query(X, k=1)

    # Compute RMS error before start iterations
    # RMS error is computed based on Neearest Neighbors
    rms_before_transf = mean_squared_error(np.vstack(ref[index]), X)
    print('The RMS (root mean square) error before ICP iteration N ({:d}) is: ({:.3f})'.format(0, rms_before_transf))
    fig, ax = plt.subplots()

    for iteration in range(max_iter):

        # Query for k-nearest neighbors (k = 1), so the nearest neighbor
        _, index = ref_KDTree.query(X, k=1)

        # Applying best_rigid_transform in order to find R and T
        R, T = best_rigid_transform(X.T, np.vstack(ref[index]).T)

        # Applying the transformation
        X = (np.dot(R, X.T) + T).T

        # Adding the Rotation, Traslation and Indexes to the lists

        R, T = calculate_global_matrices(R_prev, T_prev, R, T)

        R_prev = R
        T_prev = T
        # Computing the RMS error
        rms_error =  mean_squared_error(np.vstack(ref[index]), X)
        print('The RMS (root mean square) error after ICP iteration N ({:d}) is: ({:.3f})'.format(iteration, rms_error))

        if rms_error < RMS_threshold:
            break

        plt.cla()
        ax.plot(X.T[0], X.T[1], 'r.')
        ax.plot(ref.T[0], ref.T[1], 'b.')
        ax.set_title('ICP matching 2D')
        ax.set_title('Iteration {:d}  RMS error {:0.4f}'.format(iteration, rms_error))
        plt.axis('equal')
        plt.draw()
        plt.waitforbuttonpress(0) # this will wait for indefinite time

    plt.show()


    # Variable for aligned X, this is the cloud after applying ICP
    X_aligned = np.copy(X)

    return X_aligned, R, T

def create_points(n_points):
    #for i in range(n_points):
    data = np.random.rand(n_points, 2)*100

    theta = np.radians(-30)
    c, s = np.cos(theta), np.sin(theta)

    H = [[c, -s, 15],
         [s, c, 5],
         [0, 0, 2]]

    R = np.array([ [H[0][0], H[0][1]], [H[1][0], H[1][1]] ])
    T = np.array([ [H[0][2]], [H[1][2]] ])

    template = (np.dot(R, data.T).T + T.T)

    return data, template


if __name__ == '__main__':

    points_ref2, points2 = create_points(500)

    # Create and save data
    np.save('points_ref2.npy', points_ref2)
    np.save('points2.npy', points2)

    # # Load data
    points_ref = np.load('points_ref2.npy')
    points = np.load('points2.npy')

    # Apply ICP
    max_iter = 100
    RMS_threshold = 0.01
    data_aligned, R, T= icp_matching(points, points_ref, max_iter, RMS_threshold)