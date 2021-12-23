import numpy as np
from numpy.lib.shape_base import _column_stack_dispatcher
import scipy.linalg as la
import random
from math import pi, sin, cos
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd

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

    # Choose random 5 ECEF for chosen satellites
    mixed_ECEF = random.sample(ECEF_list, len(ECEF_list))
    chose_sat_ECEF = mixed_ECEF[:num_SVs]
    reserve_sat_ECEF = mixed_ECEF[num_SVs:]

    return chose_sat_ECEF, reserve_sat_ECEF


def gen_flight_data(num_SVs, ENU_cfp, ENU_cfp_ECEF):
    '''
    This function loads in pseudoranges and satellite ENU data for the 7 different satellites.
    It also loads in the truth data of the aircraft as well.

    Args:
        

    Returns:
        
    '''
    # Read in L1 pseudoranges for satellites (16,22,9,4,31,26,3)
    GPS_PR = pd.read_csv('./L1_L2_Sats.csv', usecols=[1,6,11,16,21,26,31]).to_numpy()

    # Read in each ENU satellite position data
    GPS16 = pd.read_csv('./L1_L2_Sats.csv', usecols=[3,4,5]).to_numpy()
    GPS22 = pd.read_csv('./L1_L2_Sats.csv', usecols=[8,9,10]).to_numpy()
    GPS9 = pd.read_csv('./L1_L2_Sats.csv', usecols=[13,14,15]).to_numpy()
    GPS4 = pd.read_csv('./L1_L2_Sats.csv', usecols=[18,19,20]).to_numpy()
    GPS31 = pd.read_csv('./L1_L2_Sats.csv', usecols=[23,24,25]).to_numpy()
    GPS26 = pd.read_csv('./L1_L2_Sats.csv', usecols=[28,29,30]).to_numpy()
    GPS3 = pd.read_csv('./L1_L2_Sats.csv', usecols=[33,34,35]).to_numpy()

    ## Read in truth data
    # Read in ENU aircraft position data
    AC_ENU = pd.read_csv('./L1_L2_Sats.csv', usecols=[36,37,38]).to_numpy()
    # Read in aircraft velocity data (NED) and convert to ENU
    cols_used = ['velocity_north','velocity_east','velocity_down']
    col_reorder = ['velocity_east','velocity_north','velocity_down']
    AC_vel = pd.read_csv('./GNSS_PLANE_wed-flight3.csv', usecols=cols_used)[col_reorder].to_numpy()
    # Change from 5hz of data to 1hz of data to match aircraft ENU data
    AC_vel = AC_vel[4:7560:5]
    # Convert DOWN into UP
    AC_vel[:, 2] *= -1

    # Read in Geodetic coordinates 
    AC_GD = pd.read_csv('./GNSS_PLANE_wed-flight3.csv', usecols=[6,7,8]).to_numpy()
    # Change from 5hz of data to 1hz of data to match aircraft ENU data
    AC_GD = AC_GD[4:7560:5]
    
    # Convert AC position from ENU to ECEF
    AC_ECEF = enu2ecef_pos(AC_ENU, ENU_cfp, ENU_cfp_ECEF)
    
    # Convert Satellite positions from ENU to ECEF
    GPS16_ECEF = enu2ecef_pos(GPS16, ENU_cfp, ENU_cfp_ECEF)
    GPS22_ECEF = enu2ecef_pos(GPS22, ENU_cfp, ENU_cfp_ECEF)
    GPS9_ECEF = enu2ecef_pos(GPS9, ENU_cfp, ENU_cfp_ECEF)
    GPS4_ECEF = enu2ecef_pos(GPS4, ENU_cfp, ENU_cfp_ECEF)
    GPS31_ECEF = enu2ecef_pos(GPS31, ENU_cfp, ENU_cfp_ECEF)
    GPS26_ECEF = enu2ecef_pos(GPS26, ENU_cfp, ENU_cfp_ECEF)
    GPS3_ECEF = enu2ecef_pos(GPS3, ENU_cfp, ENU_cfp_ECEF)

    # Convert AC velocity ENU to ECEF
    AC_vel_ECEF = enu2ecef_vel(AC_vel, ENU_cfp)

    # Pull timestamp from data to produce dt
    GPS_time = pd.read_csv('./L1_L2_Sats.csv', usecols=[0]).to_numpy()
    # Create dt
    AC_dt = np.zeros(len(GPS_time))
    for n in range(len(GPS_time)-1):
        AC_dt[n] = GPS_time[n+1] - GPS_time[n]

    # Create truth matrix
    truth_table = np.zeros((len(AC_ECEF), 8))
    
    print('hello')
        

def enu2ecef_vel(AC_vel, ENU_cfp):
    
    AC_ECEF_vel = np.zeros((len(AC_vel), 3))
    for i in range(len(AC_vel)):
        ECEF_conv = np.array([[-sin(ENU_cfp[1]), -sin(ENU_cfp[0])*cos(ENU_cfp[1]), cos(ENU_cfp[0])*cos(ENU_cfp[1])],
                            [cos(ENU_cfp[1]), -sin(ENU_cfp[0])*sin(ENU_cfp[1]), cos(ENU_cfp[0])*sin(ENU_cfp[1])],
                            [0.0, cos(ENU_cfp[0]), sin(ENU_cfp[0])]])

        ENU_vel = AC_vel[i]
        ECEF_vel = ECEF_conv.dot(ENU_vel)
        AC_ECEF_vel[i, :] = ECEF_vel 
    
    return AC_ECEF_vel

def enu2ecef_pos(ENU_data, ENU_cfp, ENU_cfp_ECEF):
    
    ECEF_data = np.zeros((len(ENU_data), 3))
    for i in range(len(ENU_data)):
        ECEF_conv = np.array([[-sin(ENU_cfp[1]), -sin(ENU_cfp[0])*cos(ENU_cfp[1]), cos(ENU_cfp[0])*cos(ENU_cfp[1])],
                            [cos(ENU_cfp[1]), -sin(ENU_cfp[0])*sin(ENU_cfp[1]), cos(ENU_cfp[0])*sin(ENU_cfp[1])],
                            [0.0, cos(ENU_cfp[0]), sin(ENU_cfp[0])]])

        ENU_coord = ENU_data[i]
        ECEF_coords = ECEF_conv.dot(ENU_coord) + ENU_cfp_ECEF 
        ECEF_data[i, :] = ECEF_coords 
    
    return ECEF_data

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

def gen_sensor_meas(num_coords, chose_sat_ECEF, truth, s_dt, Cdt):
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
    
    # Add random bias to satellite for anomaly (m)
    sat_bias = 50
    # How many secs/meas you want to add random bias to
    num_bias = 10

    # Pseudoranges from the chosen ECEF
    meas = np.zeros((len(usr_ECEF), len(chose_sat_ECEF)))
    for i, usr_pos in enumerate(usr_ECEF):
        for n, sat_pos in enumerate(chose_sat_ECEF):
            meas[i,n] = np.sqrt((sat_pos[0] - usr_pos[0])**2 + (sat_pos[1] - usr_pos[1])**2 + (sat_pos[2] - usr_pos[2])**2) + Cdt


    # Pick random spot in meas data and add satellite bias to
    bias_sat = random.randint(0, len(chose_sat_ECEF)-1)
    bias_sec = random.randint(0, len(meas)-num_bias)
    meas[bias_sec:bias_sec+(num_bias-1), bias_sat] = meas[bias_sec:bias_sec+(num_bias-1), bias_sat] + sat_bias

    # Print where the anomally will be
    print(f'Satellite w/ anomally: {bias_sat}')
    print(f'Where the anomally starts in timestep: {bias_sec}')


    return usr_ECEF, meas

def EKF(chose_sat_ECEF, sens_meas, curr_x, curr_P, Q, R):
    '''
    This function handles the EKF process of the RAIM and returns the H matrix (meas matrix) and residuals

    Args:
        sat_ECEF: an (num sat,3) array of ECEF coordinates of satellites
        sens_meas: an (num truth/s_dt, num sat) array of psuedorange measurements
        curr_x: an (8,) array of the users current state
        curr_P: an (8,8) array of the users current covariance
        Q: an (8,8) array for state error covariance matrix
        R: an (num sat, num sat) array for measurement covariance matrix

    Returns:
        curr_x: an (8,) array of the users current state (updated)
        curr_P: an (8,8) array of the users current covariance (updated)
        K: an (8, num sat) array of the Kalman Gain
        H: an (num sat, 8) array of the (H matrix) measurement matrix
        res: an (num sat,) array of the residuals of the pseudorange measurements
    '''
    ## Propagation
    # Build F Matrix (State Matrix)
    F = np.zeros((len(curr_x),len(curr_x)))
    Fcv = np.array([[1.,dt], 
                    [0.,1.]])

    for m in range(int(len(F)/2)):
        F[2*m:2*m+2,2*m:2*m+2] = Fcv


    # Build H Matrix (Measurement Matrix)
    H = np.zeros((num_SVs, len(curr_x)))
    for cnt, sat_pos in enumerate(chose_sat_ECEF):
        part_x = -(sat_pos[0] - curr_x[0]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_y = -(sat_pos[1] - curr_x[2]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_z = -(sat_pos[2] - curr_x[4]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_cdt = 1.

        H[cnt,0] = part_x
        H[cnt,2] = part_y
        H[cnt,4] = part_z
        H[cnt,6] = part_cdt
    
    # Prediction of state and covariance
    curr_x = F.dot(curr_x)
    curr_P = F.dot(curr_P).dot(F.T) + Q

    # Kalman Gain
    K = (curr_P.dot(H.T)).dot(la.inv(H.dot(curr_P).dot(H.T) + R))

    # Predicted Pseudorange Measurement (h(x) formula)
    pred_meas = np.zeros(len(chose_sat_ECEF))
    for n, sat_pos in enumerate(chose_sat_ECEF):
        pred_meas[n] = np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2) + Cdt


    # #Test info
    # p0 = np.eye(8)
    # S0 = p0.dot(H.T)
    # alp_T = np.array([0,0,1,0,0,0,0,0])
    # s0_T = alp_T.dot(S0)
    
    # Residuals (eq 26/eq 32 but using hx formula rather than Hx)
    res = sens_meas - pred_meas

    # The identity matrix for the first part of residual covariance (for meas covariance)
    I_pt1 = np.eye(len(R))

    # The identity matrix for the second part of residual covariance (for state pred covariance)
    I_pt2 = np.eye(len(P0))

    # Residual covariance matrix (Kalman Filter-Based Integrity Monitoring Against Sensor Faults Mathieu Joerger [eq 28])
    res_cov = (I_pt1 - H.dot(K)).dot(R).dot(I_pt1 - (H.dot(K))).T - H.dot(I_pt2 - K.dot(H)).dot(P0).dot((I_pt2 - K.dot(H)).T).dot(H.T)

    # Find alpha for noncentral chi square threshold (Equation 30)
    inv_R = la.inv(R)
    MeasErrCovSqrtInv = la.sqrtm(inv_R)
    SqrtResCov = la.sqrtm(res_cov)


    [U, s, V] = np.linalg.svd(MeasErrCovSqrtInv * SqrtResCov)

    alpha = s

    ## Normalize Residual
    # Standard Deviation of State
    diag_sqrtres = SqrtResCov.diagonal()
    norm_res = res / diag_sqrtres
    
    # Weighted Normal of Residual (Equation 33)
    wtd_norm_res = (norm_res.T).dot(inv_R).dot(norm_res)

    return curr_x, curr_P, K, H, alpha, res, wtd_norm_res, pred_meas, diag_sqrtres

def RAIM_chi2_global(res, res_win):
    '''
    This function handles the RAIM chi2 cumulative test statistic to verify statistic is
    within the threshold for Cumulative KF Test statistic

    Args:
        curr_x: an (8,) array of the users current state
        curr_P: an (8,8) array of the users current covariance
        R: an (num sat, num sat) array for measurement covariance matrix
        H: an (num sat, 8) array of the (H matrix) measurement matrix
        res: an (num sat,) array of the residuals of the pseudorange measurements

    Returns:
        res_list:
        cum_thres:
        thres:
    '''
    # Set window batch size
    win_size = 10
    res_win = np.asarray(res_win)

    # If array is bigger than win size delete earliest entry
    if len(res_win) > win_size:
        res_win = np.delete(res_win, 0)

    # Cumulative weighted norm residuals
    cum_res = sum(res_win)

    ## Test statistic (RAIM Portion)
    # Finding Threshold and setting Probability false alarm (Threshold found in article Weighted RAIM for Precision Approach)
    Pfa = 10e-4

    # Find inverse chi squared for threshold (m)
    thres = st.chi2.isf(q = 1-Pfa, df=(num_SVs*win_size))
    # thres = 1.0

    # spoofed_sat = np.argmax(res)
    # spoofed_sat_res = np.max(np.absolute(res))
    # print(f'Satellite {spoofed_sat} is invalid')
    # print(f'{spoofed_sat_res} m off')
    # print('\n')
        
    # Convert back to list to use append
    print(type(res_win))
    res_list = res_win.tolist()
    return res_list, cum_res, thres

def local_seq_test(i, reserve_sat_ECEF, chose_sat_ECEF, usr_ECEF, Cdt, sens_meas_mat, spoofed_sat):
    '''
    This function handles the RAIM sequential local test that will sequentially verify each 
    of the original chosen satellites to detect and exclude the satellite that has a bias.

    Args:
        i: an int that shows the timestep/number of the coordinate/user ECEF currently
        reserve_sat_ECEF: an (n,3) array of the reserved satellites position that weren't originally chosen
        usr_ECEF: a (num_coords,3) array of user position
        Cdt: set clock error 
        pred_meas: an (num sat,) array of predicted pseudoranges to find residuals
        diag_sqrtres: an (num sat,) array of the standard deviation of the state (diagonal of sqrt residuals)
        res_win: an (>=10,) array of the residuals of the pseudorange measurements based on a chosen window size

    Returns:
        nothing
    '''

    # Pull a satellite from reserved satellite list and add to chose list
    res_sat = reserve_sat_ECEF[0]
    del reserve_sat_ECEF[0]
    chose_sat_ECEF[spoofed_sat] = res_sat

    # Find Pseudorange for reserve satellites
    res_meas_cnt = len(usr_ECEF) - i
    res_meas = np.zeros(res_meas_cnt)
    for n in range(len(res_meas)):
        # Pseudoranges from the chosen ECEF
        usr_pos = usr_ECEF[i+n, :]
        res_meas[n] = np.sqrt((res_sat[0] - usr_pos[0])**2 + (res_sat[1] - usr_pos[1])**2 + (res_sat[2] - usr_pos[2])**2) + Cdt

    # Replace biased satellite measurements with new reserved measurements
    sens_meas_mat[i:, spoofed_sat] = res_meas
        
    return chose_sat_ECEF, sens_meas_mat

def plot_pseudo(sens_meas_mat, pred_mat, num_coords, s_dt):
    t = np.arange(0, num_coords, s_dt)

    #Satellite 1 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Sat 1')
    plt.plot(t, sens_meas_mat[:,0], label = "Truth")
    plt.plot(t, pred_mat[:,0], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (m)')
    plt.legend()

    #Satellite 2 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Sat 2')
    plt.plot(t, sens_meas_mat[:,1], label = "Truth")
    plt.plot(t, pred_mat[:,1], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (m)')
    plt.legend()

    #Satellite 3 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Sat 3')
    plt.plot(t, sens_meas_mat[:,2], label = "Truth")
    plt.plot(t, pred_mat[:,2], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (m)')
    plt.legend()

    #Satellite 4 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Sat 4')
    plt.plot(t, sens_meas_mat[:,3], label = "Truth")
    plt.plot(t, pred_mat[:,3], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (m)')
    plt.legend()

    #Satellite 5 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Sat 5')
    plt.plot(t, sens_meas_mat[:,4], label = "Truth")
    plt.plot(t, pred_mat[:,4], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (m)')
    plt.legend()

    plt.show()

    return

def plot_coords(truth_mat, est_state_mat):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(truth_mat[:,0], truth_mat[:,2], truth_mat[:,4], label='Truth')
    ax.scatter3D(est_state_mat[:,0], est_state_mat[:,2], est_state_mat[:,4], label='Pred')
    ax.set_title('User coordinates (ECEF)')
    ax.set_xlabel('x-coords (m)')
    ax.set_ylabel('y-coords (m)')
    ax.set_zlabel('z-coords (m)')
    ax.legend(loc='best')
    
    plt.show()
    return

def plot_res(thres_mat, cum_res_mat, num_coords, s_dt):
    t = np.arange(0, num_coords, s_dt)

    #Plotting residual and threshold each timestep
    plt.figure()
    plt.title('Cumulative Residual vs Time')
    plt.plot(t, cum_res_mat[:], label = "Cum Residual")
    plt.plot(t, thres_mat[:], label = "Threshold")
    plt.xlabel('Time (secs)')
    plt.ylabel('Cumulative Residual (m)')
    plt.legend()

    plt.show()

    return

## SET UP
# number of satellites
num_SVs = 5
# ENU Center fixed point (Geodetic Coords in radians)
ENU_cfp = np.array([0.68667326519, -1.5011475667, 192.00])
# ENU Center fixed point (ECEF km) matlab conversion
ENU_cfp_ECEF = np.array([343744.28, -4927413.92, 4022001.33])
# satellite timestep (update rate)
s_dt = 1
# number of coordinates/steps from user not including initial
num_coords = 100
# user timestep
dt = 1.0
# user constant velocity/starting velocity (m/s)
usr_vel = 25.0
# user starting ECEF coordinate (m)
usr_x0 = np.array([-681000, -4925000, 3980000])
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
chose_sat_ECEF, reserve_sat_ECEF = gen_sat_ecef(num_SVs)
gen_flight_data(num_SVs, ENU_cfp, ENU_cfp_ECEF)
truth_mat = gen_truth(num_coords, x0, dt)
usr_ECEF, sens_meas_mat = gen_sensor_meas(num_coords, chose_sat_ECEF, truth_mat, s_dt, Cdt)

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
cum_res_mat = np.zeros((num_coords))
thres_mat = np.zeros((num_coords))

# Predicted Pseudorange Matrix
pred_mat = np.zeros((num_coords, num_SVs))

## Make window size for residuals
res_win = []

for i in range(num_coords):
    # Pulling one per time step
    sens_meas = sens_meas_mat[i]
    truth = truth_mat[i]

    # EKF
    curr_x, curr_P, K, H, alpha, res, wtd_norm_res, pred_meas, diag_sqrtres = EKF(chose_sat_ECEF, sens_meas, curr_x, curr_P, Q, R)

    # RAIM chi2 global statistic check
    res_win.append(wtd_norm_res)
    res_win, cum_res, thres = RAIM_chi2_global(res, res_win)

    # Check if test statistic is within chi squared model for cumulative residual
    if cum_res < thres:
        print(f'Coordinate Point {i} is valid')
        print(f'All SVs are valid')
        print('\n')
        
    else:
        spoofed_sat = np.argmax(res)
        spoofed_sat_res = np.max(np.absolute(res))
        print(f'Coordinate Point {i} is invalid')
        print(f'Satellite {spoofed_sat} issue')
        print(f'{spoofed_sat_res} m off')
        print('\n')

        # RAIM chi2 sequential local statistic check
        chose_sat_ECEF, sens_meas_mat = local_seq_test(i, reserve_sat_ECEF, chose_sat_ECEF, usr_ECEF, Cdt, sens_meas_mat, spoofed_sat)


    # Update state and covariance
    curr_x = curr_x + K.dot(res)
    curr_P = curr_P - K.dot(H).dot(curr_P)

    # # Store for plotting
    cum_res_mat[i] = cum_res
    thres_mat[i] = thres
    # est_state_mat[i+1] = curr_x
    # est_cov_mat[i+1] = curr_P
    # pred_mat[i] = pred_meas
    # res_mat[i] = res

# Plotting and Tables
# Plot Psuedoranges of measurements and predicted measurements 
# plot_pseudo(sens_meas_mat, pred_mat, num_coords, s_dt)
# Plot Truth coordinates to Predicted Coordinates
# plot_coords(truth_mat, est_state_mat)
# Plot Cumulative Residual over time
plot_res(thres_mat, cum_res_mat, num_coords, s_dt)
# Convert Residual data to CSV for Excel Table
#np.savetxt("residuals.csv", res_mat, delimiter=",")

