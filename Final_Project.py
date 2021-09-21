import numpy as np
from numpy.lib.shape_base import _column_stack_dispatcher
import scipy.linalg as la
import random
from math import pi
import scipy.stats as st
import matplotlib.pyplot as plt

def gen_sat_ecef(num_SVs):
    '''
    This function takes in an int (num of satellites) and returns an (n,3) array 
    of ECEF coordinates for each satellite.

    Args:
        num_SV: Number of satellites 

    Returns:
        sat_ECEF: an (n,3) array of ECEF coordinates of satellites
    '''

    ECEF_list = [[10e7, -5e7, 10e7],
                [-5e7, 10e7, 10e7],
                [10e7, 5e7, 10e7],
                [-10e7, -10e7, 10e7],
                [10e7, -10e7, 10e7],
                [-10e7,  20e7, 10e7],
                [20e7, 20e7, 10e7],
                [-20e7, -10e7, 10e7]]

    sat_ECEF = random.sample(ECEF_list,num_SVs)

    return sat_ECEF


def gen_truth(num_coords, x0, dt):
    '''
    This function takes in the num of coordinates, the initial state, and time step between states
    and returns an (n,8) array of the truth data of states

    Args:
        num_coords: Number of steps/coordinates of user 
        x0: initial state 
        dt: time step between each state

    Returns:
        truth: a (num_coords,8) array of truth data of user states
    '''

    # Build A Matrix (State Matrix)
    A = np.zeros((len(x0),len(x0)))
    Acv = np.array([[1.,dt], 
                    [0.,1.]])

    for i in range(int(len(A)/2)):
        A[2*i:2*i+2,2*i:2*i+2] = Acv

    truth = np.zeros((num_coords,len(x0)))
    # noise = np.array([0.000001, 0, 0.000001, 0, 0.000001, 0, 0, 0])
    curr_state = x0 
    for i in range(num_coords):
        next_state = A.dot(curr_state)
        curr_state = next_state
        truth[i,:] = curr_state
        
        
    
    return truth

def gen_meas(num_coords, sat_ECEF, truth, s_dt, Cdt):
    '''
    This function takes in an satellite coords, truth state, timing error, and satellite update speed 
    and outputs Psuedorange vector for each satellite update.

    Args:
        sat_ECEF: an (num sat,3) array of ECEF coordinates of satellites
        truth: an (n,8) array of true user state 
        s_dt: an int representing satellite update timestep/speed
        Cdt: an int representing receiver, satellite timing error 

    Returns:
        meas: an (num truth/s_dt, num sat) array of psuedorange measurements
    '''
    usr_ECEF = np.zeros((num_coords,3))
    for i in range(len(truth)):
        usr_ECEF[i,0] = truth[i*s_dt,0]
        usr_ECEF[i,1] = truth[i*s_dt,2]
        usr_ECEF[i,2] = truth[i*s_dt,4]
    
    # Add random bias to satellite for anomaly
    sat_bias = 8.0
    # How many secs/meas you want to add random bias to
    num_bias = 10

    meas = np.zeros((len(usr_ECEF), len(sat_ECEF)))
    for i, usr_pos in enumerate(usr_ECEF):
        for n, sat_pos in enumerate(sat_ECEF):
            meas[i,n] = np.sqrt((sat_pos[0] - usr_pos[0])**2 + (sat_pos[1] - usr_pos[1])**2 + (sat_pos[2] - usr_pos[2])**2) + Cdt
    
    # Pick random spot in meas data and add satellite bias to
    bias_sat = random.randint(0, len(sat_ECEF)-1)
    bias_sec = random.randint(0, len(meas)-num_bias)
    meas[bias_sec:bias_sec+(num_bias-1), bias_sat] = meas[bias_sec:bias_sec+(num_bias-1), bias_sat] + sat_bias


    return meas


def plot_pseudo(meas, pred_mat, num_coords, s_dt):
    t = np.arange(0, num_coords, s_dt)

    #Satellite 1 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Sat 1')
    plt.plot(t, meas[:,0], label = "Truth")
    plt.plot(t, pred_mat[:,0], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (km)')
    plt.legend()

    #Satellite 2 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Sat 2')
    plt.plot(t, meas[:,1], label = "Truth")
    plt.plot(t, pred_mat[:,1], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (km)')
    plt.legend()

    #Satellite 3 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Sat 3')
    plt.plot(t, meas[:,2], label = "Truth")
    plt.plot(t, pred_mat[:,2], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (km)')
    plt.legend()

    #Satellite 4 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Sat 4')
    plt.plot(t, meas[:,3], label = "Truth")
    plt.plot(t, pred_mat[:,3], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (km)')
    plt.legend()

    #Satellite 5 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Sat 5')
    plt.plot(t, meas[:,4], label = "Truth")
    plt.plot(t, pred_mat[:,4], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (km)')
    plt.legend()

    plt.show()

    return

def plot_coords(truth, est_state_mat, num_coords):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(truth[:,0], truth[:,2], truth[:,4], label='Truth')
    ax.scatter3D(est_state_mat[:,0], est_state_mat[:,2], est_state_mat[:,4], label='Pred')
    ax.set_title('User coordinates (ECEF)')
    ax.set_xlabel('x-coords (km)')
    ax.set_ylabel('y-coords (km)')
    ax.set_zlabel('z-coords (km)')
    ax.legend(loc='best')
    
    plt.show()
    return

## SET UP
# number of satellites
num_SVs = 5
# satellite timestep (update rate)
s_dt = 1
# number of coordinates/steps from user not including initial
num_coords = 100
# user timestep
dt = 1.0
# user constant velocity/starting velocity (m/s)
usr_vel = 25.0
# user starting ECEF coordinate (km)
usr_x0 = np.array([-681, -4925, 3980])
# speed of light (m/s)
C = 299792458.0
# Psuedorange bias
Cdt = 0.0
# Psuedorange bias velocity
Cdt_dot = 0.0
# Pseudorange std dev
Pr_std = 0.0
# White noise from random walk position velocity error
Sp = 5.0
# White noise from random walk clock bias error (Cdt)
Sf = 36
# White noise from random walk clock drift error (Cdt_dot)
Sg = 0.01
# Pseudorange measurement error equal variance
rho_error = 36


# Initial State and Initial Covariance
init_usr_vel = usr_vel * np.random.randn(3)
x0 = np.array([usr_x0[0], init_usr_vel[0], usr_x0[1], init_usr_vel[1], usr_x0[2], init_usr_vel[2], Cdt, Cdt_dot])
P0 = np.eye(8)

# Get sat coordinates, truth data of states and pseudorange measurements
sat_ECEF = gen_sat_ecef(num_SVs)
truth = gen_truth(num_coords, x0, dt)
meas = gen_meas(num_coords, sat_ECEF, truth, s_dt, Cdt)

# Set current state and current covariance
curr_x = x0
curr_P = P0 

# Build State Error Covariance Matrix
Qxyz = np.array([[Sp * (dt**3)/3, Sp * (dt**2)/2],  [Sp * (dt**2)/2, Sp * dt]])
Qb = np.array([[Sf*dt + (Sg * dt**3)/3, (Sg * dt**2)/2],  [(Sg * dt**2)/2, (Sg * dt)]])

Q = np.zeros((len(curr_x),len(curr_x)))
Q[:2,:2] = Qxyz
Q[2:4,2:4] = Qxyz
Q[4:6,4:6] = Qxyz
Q[6:,6:] = Qb

# Build Measurement Error Covariance Matrix
R = np.eye(num_SVs) * rho_error

# Build User Estimated States Matrix
est_state_mat = np.zeros((num_coords+1, 8))
est_state_mat[0] = curr_x

# Build Estimated Covariance Matirx
est_cov_mat = np.zeros((num_coords+1, 8, 8))
est_cov_mat[0] = curr_P

# Residual matrix
res_mat = np.zeros((num_coords, num_SVs))

# Predicted Pseudorange Matrix
pred_mat = np.zeros((num_coords, num_SVs))

for i in range(num_coords):
    ## Propagation
    # Build F Matrix (State Matrix)
    A = np.zeros((len(curr_x),len(curr_x)))
    Acv = np.array([[1.,dt], 
                    [0.,1.]])

    for m in range(int(len(A)/2)):
        A[2*m:2*m+2,2*m:2*m+2] = Acv

    # Build H Matrix (Measurement Matrix)
    H = np.zeros((num_SVs, len(curr_x)))
    for cnt, sat_pos in enumerate(sat_ECEF):
        part_x = -(sat_pos[0] - curr_x[0]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_y = -(sat_pos[1] - curr_x[2]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_z = -(sat_pos[2] - curr_x[4]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_cdt = 1.

        H[cnt,0] = part_x
        H[cnt,2] = part_y
        H[cnt,4] = part_z
        H[cnt,6] = part_cdt
    
    
    curr_x = A.dot(curr_x)
    curr_P = A.dot(curr_P).dot(A.T) + Q

    # Kalman Gain
    K = (curr_P.dot(H.T)).dot(la.inv(H.dot(curr_P).dot(H.T) + R))

    # Predicted Pseudorange Measurement
    pred_meas = np.zeros(len(sat_ECEF))
    for n, sat_pos in enumerate(sat_ECEF):
        pred_meas[n] = np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2) + Cdt

    # Residual
    res = meas[i] - pred_meas

    ## Test statistic (RAIM Portion)
    # Variance Covariance Matrix (inverse of weight matrix)
    VCM = la.inv(H.dot(curr_P).dot(H.T) + R)
    # P matrix variance and the r.TP^-1r (inverse scales and makes independent & process change std dev)
    test_stat = (res.T).dot(VCM).dot(res)
    # Finding Threshold and setting Probability false alarm (Threshold found in article Weighted RIAM for Precision Approach)
    Pfa = 10e-4

    # Find inverse chi squared for threshold
    thres = st.chi2.isf(q = 1-Pfa, df=num_SVs)

    # Check if test statistic is within chi squared model
    if test_stat <= thres:
        print(f'Coordinate Point {i} is valid')
        
    else:
        print(f'Coordinate Point {i} is invalid')
        print(f'Threshold: {thres},  Test Statistic: {test_stat}')

    # Update state and covariance
    curr_x = curr_x + K.dot(res)
    curr_P = curr_P - K.dot(H).dot(curr_P)

    # Store for plotting
    est_state_mat[i+1] = curr_x
    est_cov_mat[i+1] = curr_P
    pred_mat[i] = pred_meas
    res_mat[i] = res

# Plotting and Tables
# Plot Psuedoranges of measurements and predicted measurements 
plot_pseudo(meas, pred_mat, num_coords, s_dt)
# Plot Truth coordinates to Predicted Coordinates

plot_coords(truth, est_state_mat, num_coords)
# Convert Residual data to CSV for Excel Table
np.savetxt("residuals.csv", res_mat, delimiter=",")


