import numpy as np
from numpy.lib.shape_base import _column_stack_dispatcher
import scipy.linalg as la
import random
from math import pi, sin, cos, sqrt
from scipy.linalg.basic import _validate_args_for_toeplitz_ops
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


def gen_flight_data(num_SVs, ENU_cfp, ENU_cfp_ECEF, Cdt, Cdt_dot):
    '''
    This function loads in pseudoranges and satellite ENU data for the 7 different satellites.
    It also loads in the truth data of the aircraft as well.

    Args:
        

    Returns:
        
    '''
    # Read in L1 pseudoranges for satellites (16,22,9,4,31,26,3)
    GPS_PR = pd.read_csv('./L1_L2_Sats.csv', usecols=[1,6,11,16,21,26,31]).to_numpy()
    # Pull out initial psuedorange
    GPS_PR_0 = GPS_PR[0,:]
    # Remove initial psuedorange
    GPS_PR = GPS_PR[1::]

    # Read in each ENU satellite position data and pull out the initial locations
    GPS16 = pd.read_csv('./L1_L2_Sats.csv', usecols=[3,4,5], skiprows=1).to_numpy()
    GPS22 = pd.read_csv('./L1_L2_Sats.csv', usecols=[8,9,10], skiprows=1).to_numpy()
    GPS9 = pd.read_csv('./L1_L2_Sats.csv', usecols=[13,14,15], skiprows=1).to_numpy()
    GPS4 = pd.read_csv('./L1_L2_Sats.csv', usecols=[18,19,20], skiprows=1).to_numpy()
    GPS31 = pd.read_csv('./L1_L2_Sats.csv', usecols=[23,24,25], skiprows=1).to_numpy()
    GPS26 = pd.read_csv('./L1_L2_Sats.csv', usecols=[28,29,30], skiprows=1).to_numpy()
    GPS3 = pd.read_csv('./L1_L2_Sats.csv', usecols=[33,34,35], skiprows=1).to_numpy()
   
    GPS_pos0 = pd.read_csv('./L1_L2_Sats.csv', usecols=[3,4,5,8,9,10,13,14,15,18,19,20,23,24,25,28,29,30,33,34,35], nrows=1).to_numpy()
    GPS_pos0 = GPS_pos0.reshape((7,3))

    # Convert Satellite positions from ENU to ECEF
    GPS16_ECEF = enu2ecef_pos(GPS16, ENU_cfp, ENU_cfp_ECEF)
    GPS22_ECEF = enu2ecef_pos(GPS22, ENU_cfp, ENU_cfp_ECEF)
    GPS9_ECEF = enu2ecef_pos(GPS9, ENU_cfp, ENU_cfp_ECEF)
    GPS4_ECEF = enu2ecef_pos(GPS4, ENU_cfp, ENU_cfp_ECEF)
    GPS31_ECEF = enu2ecef_pos(GPS31, ENU_cfp, ENU_cfp_ECEF)
    GPS26_ECEF = enu2ecef_pos(GPS26, ENU_cfp, ENU_cfp_ECEF)
    GPS3_ECEF = enu2ecef_pos(GPS3, ENU_cfp, ENU_cfp_ECEF)
    GPS_pos0_ECEF = enu2ecef_pos(GPS_pos0, ENU_cfp, ENU_cfp_ECEF)
    
    # Make initial GPS position matrix and GPS location matrix based on timestamp
    GPS_pos_matrix = np.zeros((len(GPS_pos0_ECEF)*len(GPS16_ECEF), len(GPS_pos0_ECEF[0])))
    for i in range(len(GPS16_ECEF)):
        GPS_pos_matrix[i*7,:] = GPS16_ECEF[i,:]
        GPS_pos_matrix[i*7+1,:] = GPS22_ECEF[i,:]
        GPS_pos_matrix[i*7+2,:] = GPS9_ECEF[i,:]
        GPS_pos_matrix[i*7+3,:] = GPS4_ECEF[i,:]
        GPS_pos_matrix[i*7+4,:] = GPS31_ECEF[i,:]
        GPS_pos_matrix[i*7+5,:] = GPS26_ECEF[i,:]
        GPS_pos_matrix[i*7+6,:] = GPS3_ECEF[i,:]

    ## Read in truth data
    # Read in ENU aircraft position data
    AC_ENU = pd.read_csv('./L1_L2_Sats.csv', usecols=[36,37,38]).to_numpy()
    # Read in Geodetic coordinates 
    AC_GD = pd.read_csv('./GNSS_PLANE_wed-flight3.csv', usecols=[6,7,8]).to_numpy()
    # Change from 5hz of data to 1hz of data to match aircraft ENU data
    AC_GD = AC_GD[4:7560:5]
    # Convert AC position from ENU to ECEF
    AC_ECEF = enu2ecef_pos(AC_ENU, ENU_cfp, ENU_cfp_ECEF)

    # Read in aircraft velocity data (NED) and convert to ENU
    cols_used = ['velocity_north','velocity_east','velocity_down']
    col_reorder = ['velocity_east','velocity_north','velocity_down']
    AC_vel = pd.read_csv('./GNSS_PLANE_wed-flight3.csv', usecols=cols_used)[col_reorder].to_numpy()
    # Change from 5hz of data to 1hz of data to match aircraft ENU data
    AC_vel = AC_vel[4:7560:5]
    # Convert DOWN into UP
    AC_vel[:, 2] *= -1
    # Convert AC velocity ENU to ECEF
    AC_vel_ECEF = enu2ecef_vel(AC_vel, ENU_cfp)

    # Pull timestamp from data to produce dt
    GPS_time = pd.read_csv('./L1_L2_Sats.csv', usecols=[0]).to_numpy()
    # Create dt
    AC_dt = np.zeros(len(GPS_time)-1)
    for n in range(len(GPS_time)-1):
        AC_dt[n] = GPS_time[n+1] - GPS_time[n]

    ## Create truth matrix
    # Pull out first state x0
    AC_x0 = np.array([AC_ECEF[0,0], AC_vel_ECEF[0,0], AC_ECEF[0,1], AC_vel_ECEF[0,1], AC_ECEF[0,2], AC_vel_ECEF[0,2], Cdt, Cdt_dot])
    # Remove initial position and velocity from array
    AC_ECEF = np.delete(AC_ECEF, 0, 0)
    AC_vel_ECEF = np.delete(AC_vel_ECEF, 0, 0)
    # Make truth matrix
    truth_table = np.zeros((len(AC_ECEF), 8))
    for i in range(len(truth_table)):
        truth_table[i, :] = [AC_ECEF[i,0], AC_vel_ECEF[i,0], AC_ECEF[i,1], AC_vel_ECEF[i,1], AC_ECEF[i,2], AC_vel_ECEF[i,2], Cdt, Cdt_dot]
    
    return GPS_PR_0, GPS_PR, GPS_pos0_ECEF, GPS_pos_matrix, AC_dt, AC_x0, truth_table
        

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
    bias_sec = random.randint(10, len(meas)-num_bias)
    meas[bias_sec:bias_sec+num_bias, bias_sat] = meas[bias_sec:bias_sec+num_bias, bias_sat] + sat_bias

    # Print where the anomally will be
    print(f'Satellite w/ anomally: {bias_sat}')
    print(f'Where the anomally starts in timestep: {bias_sec}')


    return usr_ECEF, meas, bias_sat, bias_sec

def EKF(chose_sat_ECEF, sens_meas, dt, curr_x, curr_P, Q, R):
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

    # Prediction of state and covariance
    curr_x = F.dot(curr_x)
    curr_P = F.dot(curr_P).dot(F.T) + Q

    # Build H Matrix (Measurement Matrix)
    H = np.zeros((len(chose_sat_ECEF), len(curr_x)))
    for cnt, sat_pos in enumerate(chose_sat_ECEF):
        part_x = -(sat_pos[0] - curr_x[0]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_y = -(sat_pos[1] - curr_x[2]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_z = -(sat_pos[2] - curr_x[4]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_cdt = 1.

        H[cnt,0] = part_x
        H[cnt,2] = part_y
        H[cnt,4] = part_z
        H[cnt,6] = part_cdt

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

    # Residual Covariance Matrix
    res_cov = H.dot(curr_P).dot(H.T) + R

    # Kalman Gain
    K = (curr_P.dot(H.T)).dot(la.inv(res_cov))

    # The identity matrix for the first part of residual covariance (for meas covariance)
    I_pt1 = np.eye(len(R))

    # The identity matrix for the second part of residual covariance (for state pred covariance)
    I_pt2 = np.eye(len(P0))

    # Residual covariance matrix (Kalman Filter-Based Integrity Monitoring Against Sensor Faults Mathieu Joerger [eq 28]) / doesn't give positive-definite 
    # res_cov = (I_pt1 - H.dot(K)).dot(R).dot(I_pt1 - (H.dot(K))).T - H.dot(I_pt2 - K.dot(H)).dot(P0).dot((I_pt2 - K.dot(H)).T).dot(H.T)

    # Take Cholesky of residual covariance
    # Equation given by Dr.Leishman from matlab code
    L_res_cov = np.linalg.cholesky(res_cov)

    # Find alpha for noncentral chi square threshold (Equation 30)
    # inv_R = la.inv(R)
    # MeasErrCovSqrtInv = la.sqrtm(inv_R)
    # SqrtResCov = la.sqrtm(res_cov)


    # [U, s, V] = np.linalg.svd(MeasErrCovSqrtInv * SqrtResCov)

    # alpha = s

    ## Normalize Residual
    # Standard Deviation of State
    # diag_sqrtres = SqrtResCov.diagonal()
    # Equation given by Dr.Leishman from matlab code
    norm_res = la.inv(L_res_cov).dot(res)
    
    # Weighted Normal of Residual (Equation 33)
    # wtd_norm_res = (norm_res.T).dot(inv_R).dot(norm_res)
    # Equation given by Dr.Leishman from matlab code
    wtd_norm_res = np.inner(norm_res.T,norm_res)

    return curr_x, curr_P, K, H, res, wtd_norm_res, pred_meas

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
    # thres = st.chi2.isf(q = 1-Pfa, df=(num_SVs*win_size))
    thres = 50.00

    # spoofed_sat = np.argmax(res)
    # spoofed_sat_res = np.max(np.absolute(res))
    # print(f'Satellite {spoofed_sat} is invalid')
    # print(f'{spoofed_sat_res} m off')
    # print('\n')
        
    # Convert back to list to use append
    print(type(res_win))
    res_list = res_win.tolist()
    return res_list, cum_res, thres

def local_seq_test(curr_x, sens_meas, rho_error, i, reserve_sat_ECEF, chose_sat_ECEF, usr_ECEF, Cdt, sens_meas_mat, pred_mat, spoofed_sat):
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
    # Remove faulty satellite 
    del chose_sat_ECEF[spoofed_sat]
    # Remove sensor pseudorange measurement from faulty satellite
    sens_meas = np.delete(sens_meas, spoofed_sat)
    # Change Measurement Covariance Matrix to a 4x4
    R = np.eye(len(chose_sat_ECEF))*rho_error

    # Refind H matrix (Measurement Matrix) w/o the faulty matrix 
    H = np.zeros((len(chose_sat_ECEF), len(curr_x)))
    for cnt, sat_pos in enumerate(chose_sat_ECEF):
        part_x = -(sat_pos[0] - curr_x[0]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_y = -(sat_pos[1] - curr_x[2]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_z = -(sat_pos[2] - curr_x[4]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2)
        part_cdt = 1.

        H[cnt,0] = part_x
        H[cnt,2] = part_y
        H[cnt,4] = part_z
        H[cnt,6] = part_cdt

    # Refind the Predicted Pseudorange Measurement (h(x) formula) w/o the faulty matrix 
    pred_meas = np.zeros(len(chose_sat_ECEF))
    for n, sat_pos in enumerate(chose_sat_ECEF):
        pred_meas[n] = np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[2])**2 + (sat_pos[2] - curr_x[4])**2) + Cdt

    # Refind Residuals (eq 26/eq 32 but using hx formula rather than Hx) w/o the faulty matrix 
    res = sens_meas - pred_meas

    # Refind Residual Covariance Matrix w/o the faulty matrix 
    res_cov = H.dot(curr_P).dot(H.T) + R

    # Refind Kalman Gain w/o the faulty matrix 
    K = (curr_P.dot(H.T)).dot(la.inv(res_cov))

    # Pull a satellite from reserved satellite list and add to chose list
    res_sat = reserve_sat_ECEF[0]
    del reserve_sat_ECEF[0]
    chose_sat_ECEF.append(res_sat)

    # Find Pseudorange for reserve satellite and add it to end of matrix
    res_meas = np.zeros(len(sens_meas_mat))
    for n in range(len(res_meas)):
        # Pseudoranges from the chosen ECEF
        usr_pos = usr_ECEF[n, :]
        res_meas[n] = np.sqrt((res_sat[0] - usr_pos[0])**2 + (res_sat[1] - usr_pos[1])**2 + (res_sat[2] - usr_pos[2])**2) + Cdt


    # Pull out biased pseudorange up til trigger point for plotting 
    bias_sat_meas_mat = sens_meas_mat[:i, spoofed_sat]

    # Pull out biased predicted up til trigger point for plot
    bias_sat_meas_pred_mat = pred_mat[:i, spoofed_sat]

    # Replace biased satellite measurements with new reserved measurements
    res_meas = res_meas.reshape((len(res_meas),1))
    sens_meas_mat = np.delete(sens_meas_mat, spoofed_sat, 1)
    sens_meas_mat = np.append(sens_meas_mat, res_meas, 1)
        
    return H, res, K, chose_sat_ECEF, sens_meas, pred_meas, sens_meas_mat, bias_sat_meas_mat, bias_sat_meas_pred_mat

def plot_pseudo(valid_sat_meas_mat, bias_sat_meas_mat, valid_pred_sat_meas_mat, bias_sat_meas_pred_mat, num_coords, s_dt):
    t = np.arange(0, num_coords, s_dt)

    # Valid Satellite 0 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Valid Sat 0')
    plt.plot(t, valid_sat_meas_mat[:,0], label = "Truth")
    plt.plot(t, valid_pred_sat_meas_mat[:,0], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (m)')
    plt.legend()

    # Valid Satellite 1 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Valid Sat 1')
    plt.plot(t, valid_sat_meas_mat[:,1], label = "Truth")
    plt.plot(t, valid_pred_sat_meas_mat[:,1], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (m)')
    plt.legend()

    # Valid Satellite 2 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Valid Sat 2')
    plt.plot(t, valid_sat_meas_mat[:,2], label = "Truth")
    plt.plot(t, valid_pred_sat_meas_mat[:,2], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (m)')
    plt.legend()

    # Valid Satellite 3 Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Valid Sat 3')
    plt.plot(t, valid_sat_meas_mat[:,3], label = "Truth")
    plt.plot(t, valid_pred_sat_meas_mat[:,3], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (m)')
    plt.legend()

    # Biased Satellite Psuedorange Plot
    plt.figure()
    plt.title('Psuedorange vs Time for Biased Sat')
    plt.plot(t, bias_sat_meas_mat, label = "Truth")
    plt.plot(t, bias_sat_meas_pred_mat, label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('Psuedorange (m)')
    plt.legend()

    plt.show()

    return

def plot_coords(truth_mat, est_state_mat, cov_bounds, num_coords, dt):

    t = np.arange(0, num_coords, dt)
    up_bound = cov_bounds
    lw_bound = cov_bounds*-1

    # Plotting Truth vs Predicted User Coords x-axis
    plt.figure()
    plt.title('User coordinates for x-axis (ECEF)')
    plt.plot(t, truth_mat[:,0], label = "Truth")
    plt.plot(t, est_state_mat[:,0], label = "Pred")
    plt.plot(t, up_bound[:,0], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,0], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords x-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User Coords y-axis
    plt.figure()
    plt.title('User coordinates for y-axis (ECEF)')
    plt.plot(t, truth_mat[:,2], label = "Truth")
    plt.plot(t, est_state_mat[:,2], label = "Pred")
    plt.plot(t, up_bound[:,2], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,2], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords y-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User Coords z-axis
    plt.figure()
    plt.title('User coordinates for z-axis (ECEF)')
    plt.plot(t, truth_mat[:,4], label = "Truth")
    plt.plot(t, est_state_mat[:,4], label = "Pred")
    plt.plot(t, up_bound[:,4], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,4], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords z-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity x-axis
    plt.figure()
    plt.title('User Velocity for x-axis (ECEF)')
    plt.plot(t, truth_mat[:,1], label = "Truth")
    plt.plot(t, est_state_mat[:,1], label = "Pred")
    plt.plot(t, up_bound[:,1], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,1], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity x-axis (m/s)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity y-axis
    plt.figure()
    plt.title('User Velocity for y-axis (ECEF)')
    plt.plot(t, truth_mat[:,3], label = "Truth")
    plt.plot(t, est_state_mat[:,3], label = "Pred")
    plt.plot(t, up_bound[:,3], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,3], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity y-axis (m/s)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity z-axis
    plt.figure()
    plt.title('User Velocity for z-axis (ECEF)')
    plt.plot(t, truth_mat[:,5], label = "Truth")
    plt.plot(t, est_state_mat[:,5], label = "Pred")
    plt.plot(t, up_bound[:,5], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,5], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity z-axis (m/s)')
    plt.legend()

    

    # Plot predicted 3D coords vs truth 3D coords
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(truth_mat[:,0], truth_mat[:,2], truth_mat[:,4], label='Truth')
    # ax.scatter3D(est_state_mat[:,0], est_state_mat[:,2], est_state_mat[:,4], label='Pred')
    # ax.set_title('User coordinates (ECEF)')
    # ax.set_xlabel('x-coords (m)')
    # ax.set_ylabel('y-coords (m)')
    # ax.set_zlabel('z-coords (m)')
    # ax.legend(loc='best')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(truth_mat[:,1], truth_mat[:,3], truth_mat[:,5], label='Truth')
    # ax.scatter3D(est_state_mat[:,1], est_state_mat[:,3], est_state_mat[:,5], label='Pred')
    # ax.set_title('User velocity (ECEF)')
    # ax.set_xlabel('x-coords (m/s)')
    # ax.set_ylabel('y-coords (m/s)')
    # ax.set_zlabel('z-coords (m/s)')
    # ax.legend(loc='best')
    
    plt.show()
    return

def plot_error(truth_mat, est_state_mat, cov_bounds, num_coords, dt):

    # Set up timestamp (x-axis)
    t = np.arange(0, num_coords, dt)
    # Make covariance upper and lower bound
    up_bound = cov_bounds
    lw_bound = cov_bounds*-1
    # Make error between truth and predicted state
    state_error = truth_mat - est_state_mat

    # Plotting Truth vs Predicted User Coords x-axis
    plt.figure()
    plt.title('User coordinates error for x-axis (ECEF)')
    plt.plot(t, state_error[:,0], label = "Error")
    plt.plot(t, up_bound[:,0], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,0], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords Error x-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User Coords y-axis
    plt.figure()
    plt.title('User coordinates error for y-axis (ECEF)')
    plt.plot(t, state_error[:,2], label = "Error")
    plt.plot(t, up_bound[:,2], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,2], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords Error y-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User Coords z-axis
    plt.figure()
    plt.title('User coordinates error for z-axis (ECEF)')
    plt.plot(t, state_error[:,4], label = "Error")
    plt.plot(t, up_bound[:,4], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,4], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords Error z-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity x-axis
    plt.figure()
    plt.title('User Velocity error for x-axis (ECEF)')
    plt.plot(t, state_error[:,1], label = "Error")
    plt.plot(t, up_bound[:,1], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,1], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity Error x-axis (m/s)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity y-axis
    plt.figure()
    plt.title('User Velocity error for y-axis (ECEF)')
    plt.plot(t, state_error[:,3], label = "Error")
    plt.plot(t, up_bound[:,3], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,3], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity Error y-axis (m/s)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity z-axis
    plt.figure()
    plt.title('User Velocity error for z-axis (ECEF)')
    plt.plot(t, state_error[:,5], label = "Error")
    plt.plot(t, up_bound[:,5], color = 'black', label = "Upper Bound")
    plt.plot(t, lw_bound[:,5], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity Error z-axis (m/s)')
    plt.legend()

    
    plt.show()
    return    

def plot_res(bias_sat, bias_sec, thres_mat, cum_res_mat, num_coords, s_dt):
    t = np.arange(0, num_coords, s_dt)

    #Plotting residual and threshold each timestep
    plt.figure()
    plt.title('Cumulative Residual vs Time | Satellite w/ Bias: ' +str(bias_sat)+ ', Bias: ' +str(bias_sec))
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
# rho_error = 36
# Needs to be std dev
rho_error = 1.0
# Random samples normal distribution for white noise
white_noise = np.random.default_rng()

# Initial State and Initial Covariance
init_usr_vel = usr_vel * np.random.randn(3)
x0 = np.array([usr_x0[0], init_usr_vel[0], usr_x0[1], init_usr_vel[1], usr_x0[2], init_usr_vel[2], Cdt, Cdt_dot])
P0 = np.eye(8)

# Set current state and current covariance
curr_x = x0
curr_P = P0 

# Get sat coordinates, truth data of states and pseudorange measurements
chose_sat_ECEF, reserve_sat_ECEF = gen_sat_ecef(num_SVs)
# Real AC data
# GPS_PR_0, GPS_PR, GPS_pos0_ECEF, GPS_pos_matrix, AC_dt, AC_x0, truth_table = gen_flight_data(num_SVs, ENU_cfp, ENU_cfp_ECEF, Cdt, Cdt_dot)
truth_mat = gen_truth(num_coords, x0, dt)
usr_ECEF, sens_meas_mat, bias_sat, bias_sec = gen_sensor_meas(num_coords, chose_sat_ECEF, truth_mat, s_dt, Cdt)

# Build State Error Covariance Matrix
Qxyz = np.array([[Sp * (dt**3)/3, Sp * (dt**2)/2],  [Sp * (dt**2)/2, Sp * dt]])
Qb = np.array([[Sf*dt + (Sg * dt**3)/3, (Sg * dt**2)/2],  [(Sg * dt**2)/2, (Sg * dt)]])

Q = np.zeros((len(curr_x),len(curr_x)))
Q[:2,:2] = Qxyz
Q[2:4,2:4] = Qxyz
Q[4:6,4:6] = Qxyz
Q[6:,6:] = Qb

# Build Measurement Error Covariance Matrix
R = np.eye(num_SVs) * rho_error**2

# Build User Estimated States Matrix
est_state_mat = np.zeros((num_coords, 8))

# Build Estimated Covariance Matirx
est_cov_mat = np.zeros((num_coords, 8, 8))

# Standard covariance for plotting bounds
cov_bounds = np.zeros((num_coords, 8))

# To store the sensor measurement and changes for plotting
sens_meas_mat_plot = np.zeros((num_coords, len(chose_sat_ECEF)))

# Residual matrix
res_mat = []
cum_res_mat = np.zeros((num_coords))
thres_mat = np.zeros((num_coords))

# Predicted Pseudorange Matrix
pred_mat = np.zeros((num_coords, len(chose_sat_ECEF)))

## Make window size for residuals
res_win = []

for i in range(num_coords):
    # Pulling one per time step
    # Adding white noise to sensor measurement
    sens_meas = sens_meas_mat[i] + white_noise.normal(0.0, rho_error, 5)
    truth = truth_mat[i]

    # EKF
    curr_x, curr_P, K, H, res, wtd_norm_res, pred_meas  = EKF(chose_sat_ECEF, sens_meas, dt, curr_x, curr_P, Q, R)

    # RAIM chi2 global statistic check
    res_win.append(wtd_norm_res)
    res_win, cum_res, thres = RAIM_chi2_global(res, res_win)

    # Check if test statistic is within chi squared model for cumulative residual
    if cum_res < thres:
        print(f'Coordinate Point {i} is valid')
        print(f'All SVs are valid')
        print('\n')
        
    else:
        spoofed_sat = np.argmax(np.absolute(res))
        spoofed_sat_res = np.max(np.absolute(res))
        print(f'Coordinate Point {i} is invalid')
        print(f'Satellites Residuals: {res}')
        print(f'Satellite {spoofed_sat} issue')
        print(f'{spoofed_sat_res} m off')
        print('\n')

        # Timestep when bias was caught
        bias_catch_ts = i
        # RAIM chi2 sequential local statistic check
        H, res, K, chose_sat_ECEF, sens_meas, pred_meas, sens_meas_mat, bias_sat_meas_mat, bias_sat_meas_pred_mat = local_seq_test(curr_x, sens_meas, rho_error, i, reserve_sat_ECEF, chose_sat_ECEF, usr_ECEF, Cdt, sens_meas_mat, pred_mat, spoofed_sat)
        # Clear residual window
        res_win.clear()
        # Add zero to end of sensor measurement and predicted measurement for plotting
        sens_meas = np.append(sens_meas, 0.0)
        pred_meas = np.append(pred_meas, 0.0)

    # Update state and covariance
    curr_x = curr_x + K.dot(res)
    curr_P = curr_P - K.dot(H).dot(curr_P)

    # Standardize Covariance for plot bounds
    std_cov = np.sqrt(curr_P.diagonal())
    
    # # Store for plotting
    res_mat.append(res)
    sens_meas_mat_plot[i] = sens_meas
    cum_res_mat[i] = cum_res
    thres_mat[i] = thres
    est_state_mat[i] = curr_x
    est_cov_mat[i] = curr_P
    cov_bounds[i] = std_cov
    pred_mat[i] = pred_meas
    

# Plotting and Tables
# Reorganize matrices to plot satellites accurately
if spoofed_sat == 0:
    valid_gps_0 = sens_meas_mat_plot[:bias_catch_ts,1]
    valid_gps_0 = np.append(valid_gps_0, sens_meas_mat_plot[bias_catch_ts:,0])
    valid_gps_1 = sens_meas_mat_plot[:bias_catch_ts,2]
    valid_gps_1 = np.append(valid_gps_1, sens_meas_mat_plot[bias_catch_ts:,1])
    valid_gps_2 = sens_meas_mat_plot[:bias_catch_ts,3]
    valid_gps_2 = np.append(valid_gps_2, sens_meas_mat_plot[bias_catch_ts:,2])
    valid_gps_3 = sens_meas_mat_plot[:bias_catch_ts,4]
    valid_gps_3 = np.append(valid_gps_3, sens_meas_mat_plot[bias_catch_ts:,3])

    pred_gps_0 = pred_mat[:bias_catch_ts,1]
    pred_gps_0 = np.append(pred_gps_0, pred_mat[bias_catch_ts:,0])
    pred_gps_1 = pred_mat[:bias_catch_ts,2]
    pred_gps_1 = np.append(pred_gps_1, pred_mat[bias_catch_ts:,1])
    pred_gps_2 = pred_mat[:bias_catch_ts,3]
    pred_gps_2 = np.append(pred_gps_2, pred_mat[bias_catch_ts:,2])
    pred_gps_3 = pred_mat[:bias_catch_ts,4]
    pred_gps_3 = np.append(pred_gps_3, pred_mat[bias_catch_ts:,3])

elif spoofed_sat == 1:
    valid_gps_0 = sens_meas_mat_plot[:,0]
    valid_gps_1 = sens_meas_mat_plot[:bias_catch_ts,2]
    valid_gps_1 = np.append(valid_gps_1, sens_meas_mat_plot[bias_catch_ts:,1])
    valid_gps_2 = sens_meas_mat_plot[:bias_catch_ts,3]
    valid_gps_2 = np.append(valid_gps_2, sens_meas_mat_plot[bias_catch_ts:,2])
    valid_gps_3 = sens_meas_mat_plot[:bias_catch_ts,4]
    valid_gps_3 = np.append(valid_gps_3, sens_meas_mat_plot[bias_catch_ts:,3])

    pred_gps_0 = pred_mat[:,0]
    pred_gps_1 = pred_mat[:bias_catch_ts,2]
    pred_gps_1 = np.append(pred_gps_1, pred_mat[bias_catch_ts:,1])
    pred_gps_2 = pred_mat[:bias_catch_ts,3]
    pred_gps_2 = np.append(pred_gps_2, pred_mat[bias_catch_ts:,2])
    pred_gps_3 = pred_mat[:bias_catch_ts,4]
    pred_gps_3 = np.append(pred_gps_3, pred_mat[bias_catch_ts:,3])

elif spoofed_sat == 2:
    valid_gps_0 = sens_meas_mat_plot[:,0]
    valid_gps_1 = sens_meas_mat_plot[:,1]
    valid_gps_2 = sens_meas_mat_plot[:bias_catch_ts,3]
    valid_gps_2 = np.append(valid_gps_2, sens_meas_mat_plot[bias_catch_ts:,2])
    valid_gps_3 = sens_meas_mat_plot[:bias_catch_ts,4]
    valid_gps_3 = np.append(valid_gps_3, sens_meas_mat_plot[bias_catch_ts:,3])

    pred_gps_0 = pred_mat[:,0]
    pred_gps_1 = pred_mat[:,1]
    pred_gps_2 = pred_mat[:bias_catch_ts,3]
    pred_gps_2 = np.append(pred_gps_2, pred_mat[bias_catch_ts:,2])
    pred_gps_3 = pred_mat[:bias_catch_ts,4]
    pred_gps_3 = np.append(pred_gps_3, pred_mat[bias_catch_ts:,3])

elif spoofed_sat == 3:
    valid_gps_0 = sens_meas_mat_plot[:,0]
    valid_gps_1 = sens_meas_mat_plot[:,1]
    valid_gps_2 = sens_meas_mat_plot[:,2]
    valid_gps_3 = sens_meas_mat_plot[:bias_catch_ts,4]
    valid_gps_3 = np.append(valid_gps_3, sens_meas_mat_plot[bias_catch_ts:,3])

    pred_gps_0 = pred_mat[:,0]
    pred_gps_1 = pred_mat[:,1]
    pred_gps_2 = pred_mat[:,2]
    pred_gps_3 = pred_mat[:bias_catch_ts,4]
    pred_gps_3 = np.append(pred_gps_3, pred_mat[bias_catch_ts:,3])

elif spoofed_sat == 4:
    valid_gps_0 = sens_meas_mat_plot[:,0]
    valid_gps_1 = sens_meas_mat_plot[:,1]
    valid_gps_2 = sens_meas_mat_plot[:,2]
    valid_gps_3 = sens_meas_mat_plot[:,3]

    pred_gps_0 = pred_mat[:,0]
    pred_gps_1 = pred_mat[:,1]
    pred_gps_2 = pred_mat[:,2]
    pred_gps_3 = pred_mat[:,3]

valid_sat_meas_mat = np.vstack((valid_gps_0,valid_gps_1,valid_gps_2,valid_gps_3)).T
valid_pred_sat_meas_mat = np.vstack((pred_gps_0,pred_gps_1,pred_gps_2,pred_gps_3)).T
bias_sat_meas_mat = np.append(bias_sat_meas_mat, sens_meas_mat_plot[bias_catch_ts:, -1])
bias_sat_meas_pred_mat = np.append(bias_sat_meas_pred_mat, pred_mat[bias_catch_ts:, -1])


# Plot Psuedoranges of measurements and predicted measurements 
plot_pseudo(valid_sat_meas_mat, bias_sat_meas_mat, valid_pred_sat_meas_mat, bias_sat_meas_pred_mat, num_coords, s_dt)
# Plot Truth coordinates to Predicted Coordinates
# plot_coords(truth_mat, est_state_mat, cov_bounds, num_coords, dt)
# Plot Error with covariance bound
plot_error(truth_mat, est_state_mat, cov_bounds, num_coords, dt)
# Plot Cumulative Residual over time
plot_res(bias_sat, bias_sec, thres_mat, cum_res_mat, num_coords, s_dt)
# Convert Residual data to CSV for Excel Table
# np.savetxt("residuals.csv", res_mat, delimiter=",")

