import numpy as np
from numpy.core.arrayprint import set_string_function
from numpy.lib.shape_base import _column_stack_dispatcher
from pandas.core.base import NoNewAttributesMixin
import scipy.linalg as la
import random
from math import nan, pi, sin, cos, sqrt
from scipy.linalg.basic import _validate_args_for_toeplitz_ops
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
from gnc import vanloan


def gen_flight_data(ENU_cfp, ENU_cfp_ECEF, Cdt):
    '''
    This function loads in pseudoranges and satellite ENU data for the 7 different satellites.
    It also loads in the truth data of the aircraft as well.

    Args:
        

    Returns:
        
    '''
    # Read in L1 pseudoranges for satellites (3,4,9,16,22,26,27,31)
    GPS_PR = pd.read_csv('./PRtrunc_resize.csv', header= None, usecols=[3]).to_numpy()

    # Pull out initial psuedorange
    GPS_PR_0 = GPS_PR[0:8,:]
    # Remove initial psuedorange
    GPS_PR = GPS_PR[8::]

    # Read in each ENU satellite position data (3,4,9,16,22,26,27,31)
    GPS_pos_matrix = pd.read_csv('./SV_ENU0trunc_resize.csv', header= None, usecols=[3,4,5]).to_numpy()

    # Pull out initial GPS positions
    GPS_pos_0 = GPS_pos_matrix[0:8,:]
    # Remove initial GPS positions
    GPS_pos_matrix = GPS_pos_matrix[8::]

    ## Read in truth data
    # Read in ENU aircraft position data
    AC_ENU = pd.read_csv('./enu_int_trunc.csv', header= None).to_numpy()

    # Read in aircraft velocity data (NED) and convert to ENU
    cols_used = ['velocity_north','velocity_east','velocity_down']
    col_reorder = ['velocity_east','velocity_north','velocity_down']
    AC_vel = pd.read_csv('./GNSS_PLANE_wed-flight3.csv', usecols=cols_used)[col_reorder].to_numpy()
    # Change from 5hz of data to 1hz of data to match aircraft ENU data
    AC_vel = AC_vel[4:7560:5]
    # Start at GPS time of week 351127 for new data (timestep 235)
    AC_vel = AC_vel[235::]
    # Insert initial velocity for initial position 
    AC_vel = np.insert(AC_vel,0,[0,0,0],axis=0)
    
    # Convert DOWN into UP
    AC_vel[:, 2] *= -1

    # Pull timestamp from data to produce dt
    GPS_time = pd.read_csv('./truth_time0_trunc.csv').to_numpy()
    # Create dt
    AC_dt = np.zeros(len(GPS_time))
    AC_dt[0] = 1.0
    for n in range(len(GPS_time)-1):
        AC_dt[n+1] = GPS_time[n+1] - GPS_time[n]

    ## Create truth table
    # Pull out first state x0
    AC_x0 = np.array([AC_ENU[0,0], AC_vel[0,0], AC_ENU[0,1], AC_vel[0,1], AC_ENU[0,2], AC_vel[0,2], 0.])
    # Remove initial position and velocity from array
    AC_ENU = np.delete(AC_ENU, 0, 0)
    AC_vel = np.delete(AC_vel, 0, 0)
    # Make truth matrix
    truth_table = np.zeros((len(AC_ENU), 7))
    for i in range(len(truth_table)):
        truth_table[i, :] = [AC_ENU[i,0], AC_vel[i,0], AC_ENU[i,1], AC_vel[i,1], AC_ENU[i,2], AC_vel[i,2], Cdt]
    
    return GPS_PR, GPS_PR_0, GPS_pos_0, GPS_pos_matrix, AC_dt, AC_x0, truth_table
        

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


def EKF(sat_ENU, sens_meas, dt, curr_x, curr_P):
    '''
    This function handles the EKF process of the RAIM and returns the H matrix (meas matrix) and residuals

    Args:
        sat_ENU: an (num sat,3) array of ENU coordinates of satellites
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

    # Constants from ARMAS EKF setup
    accelbias_sigma = 100
    accelbias_tau = 100
    clockbias_sigma = 8000
    clockbias_tau = 3600
    # clockbias_sigma = 1
    # clockbias_tau = 3600

    # Pseudorange measurement error equal variance
    # Needs to be std dev
    rho_error = 2.0

    # Build State Error Covariance Matrix
    Q = np.zeros((len(curr_x),len(curr_x)))
    Q_acc = 2.0*(accelbias_sigma**2)/accelbias_tau
    Q__cb = 2.0*(clockbias_sigma**2)/clockbias_tau
    Q_diag = [0, 0, Q_acc, 0, 0, Q_acc, 0, 0, Q_acc, Q__cb]
    np.fill_diagonal(Q, Q_diag)

    # Build Measurement Error Covariance Matrix
    R = np.eye(len(sens_meas)) * rho_error**2

    # Build Input Matrix
    B = np.zeros((len(curr_x),len(curr_x)))

    # Set Sampling Period (T) to dt for Van Loan Method
    T = dt

    ## Propagation
    # Build F Matrix (State Matrix) (No coupling between x,y,z,cdt therefore state matrix is block diagonal)

    F = np.zeros((len(curr_x),len(curr_x)))
    # Block diagonal matrix for position, velocity, and acceleration
    Fcv_pva = np.array([[0.,1.,0.], 
                    [0.,0.,1.],
                    [0.,0.,(-1/accelbias_tau)]])


    # Influence of clock bias on state
    F_cb = -1./clockbias_tau

    for m in range(int(len(F)/3)):
        F[3*m:3*m+3,3*m:3*m+3] = Fcv_pva
    F[9:,9:] = F_cb

    # Find Discretized/Linearized Matrices of F, B, and Q using Van Loan Method authored by David Woodburn in gnc.py file
    Phi, Bd, Qd = vanloan(F,B,Q,T)

    # Prediction of state and covariance
    curr_x = Phi.dot(curr_x)
    curr_P = Phi.dot(curr_P).dot(Phi.T) + Qd

    # Build H Matrix (Measurement Matrix)
    H = np.zeros((len(sat_ENU), len(curr_x)))
    for cnt, sat_pos in enumerate(sat_ENU):
        part_x = -(sat_pos[0] - curr_x[0]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[3])**2 + (sat_pos[2] - curr_x[6])**2)
        part_y = -(sat_pos[1] - curr_x[3]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[3])**2 + (sat_pos[2] - curr_x[6])**2)
        part_z = -(sat_pos[2] - curr_x[6]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[3])**2 + (sat_pos[2] - curr_x[6])**2)
        part_cdt = 1.

        H[cnt,0] = part_x
        H[cnt,3] = part_y
        H[cnt,6] = part_z
        H[cnt,9] = part_cdt

    # Predicted Pseudorange Measurement (h(x) formula)
    pred_meas = np.zeros(len(sat_ENU))
    for n, sat_pos in enumerate(sat_ENU):
        pred_meas[n] = np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[3])**2 + (sat_pos[2] - curr_x[6])**2) + Cdt

    
    # Residuals (eq 26/eq 32 but using hx formula rather than Hx)
    res = sens_meas - pred_meas

    # Residual Covariance Matrix
    res_cov = H.dot(curr_P).dot(H.T) + R

    # Kalman Gain
    K = (curr_P.dot(H.T)).dot(la.inv(res_cov))

    # Take Cholesky of residual covariance
    # Equation given by Dr.Leishman from matlab code
    L_res_cov = np.linalg.cholesky(res_cov)

    ## Normalize Residual
    # Equation given by Dr.Leishman from matlab code
    norm_res = la.inv(L_res_cov).dot(res)
    
    # Weighted Normal of Residual
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
    # Pfa (prob of false alarm) of 0.1 and Pmd (prob of missed detection) of 0.2 was used in Integrity Monitoring in Navigation Systems: Fault Detection and Exclusion RAIM Algorithm Implementation
    # In aviatioin systems Pfa is fixed to Pfa is fixed at 1/15000.
    # Pfa is 3.3*10^-7 in Ryan S., Augmentation of DGPS for Marine Navigation, Ph.D. thesis, The University of Calgary, UCGE Report 20164, 2002, 248 p
    Pfa = 10e-4

    #Wang, Wenbo, and Ying Xu. "A modified residual-based RAIM algorithm for multiple outliers based on a robust MM estimation." Sensors 20.18 (2020): 5407.
    #Test statistic T_rb = sqrt(r.T.dot(W.dot(r))/(num SV - num states- 3))
    # GNSS Signal Reliability Testing in Urban and Indoor Environments by Heidi Kuusniemi
    # Test statistic/variance factor = r.T.dot(la.inv(P)).dot(r)/(n-p) degrees of freedom
    # variance of threshold = chi-square threshold / (n-p) degrees of freedom

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

def local_seq_test(curr_x, sens_meas, rho_error, i, sat_ENU, spoofed_sat):
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
        part_x = -(sat_pos[0] - curr_x[0]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[3])**2 + (sat_pos[2] - curr_x[6])**2)
        part_y = -(sat_pos[1] - curr_x[3]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[3])**2 + (sat_pos[2] - curr_x[6])**2)
        part_z = -(sat_pos[2] - curr_x[6]) / np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[3])**2 + (sat_pos[2] - curr_x[6])**2)
        part_cdt = 1.

        H[cnt,0] = part_x
        H[cnt,2] = part_y
        H[cnt,4] = part_z
        H[cnt,6] = part_cdt

    # Refind the Predicted Pseudorange Measurement (h(x) formula) w/o the faulty matrix 
    pred_meas = np.zeros(len(chose_sat_ECEF))
    for n, sat_pos in enumerate(chose_sat_ECEF):
        pred_meas[n] = np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[3])**2 + (sat_pos[2] - curr_x[6])**2) + Cdt

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
        
    return H, res, K, sat_ENU, sens_meas, pred_meas

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

def plot_coords(num_coords, truth_table, est_state_mat, AC_dt):
    # Make timestep for plot
    timestep = np.zeros(num_coords)
    t = 0
    for i in range(len(timestep)):
        t += AC_dt[i]
        timestep[i] = t

    # Truth table should match timestep length
    truth_table = truth_table[:num_coords,:]

    # Plotting Truth vs Predicted User Coords x-axis
    plt.figure()
    plt.title('User coordinates for x-axis (ENU)')
    plt.plot(timestep, truth_table[:,0], label = "Truth")
    plt.plot(timestep, est_state_mat[:,0], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords x-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User Coords y-axis
    plt.figure()
    plt.title('User coordinates for y-axis (ENU)')
    plt.plot(timestep, truth_table[:,2], label = "Truth")
    plt.plot(timestep, est_state_mat[:,3], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords y-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User Coords z-axis
    plt.figure()
    plt.title('User coordinates for z-axis (ENU)')
    plt.plot(timestep, truth_table[:,4], label = "Truth")
    plt.plot(timestep, est_state_mat[:,6], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords z-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity x-axis
    plt.figure()
    plt.title('User Velocity for x-axis (ENU)')
    plt.plot(timestep, truth_table[:,1], label = "Truth")
    plt.plot(timestep, est_state_mat[:,1], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity x-axis (m/s)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity y-axis
    plt.figure()
    plt.title('User Velocity for y-axis (ENU)')
    plt.plot(timestep, truth_table[:,3], label = "Truth")
    plt.plot(timestep, est_state_mat[:,4], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity y-axis (m/s)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity z-axis
    plt.figure()
    plt.title('User Velocity for z-axis (ENU)')
    plt.plot(timestep, truth_table[:,5], label = "Truth")
    plt.plot(timestep, est_state_mat[:,7], label = "Pred")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity z-axis (m/s)')
    plt.legend()
    
    plt.show()
    return

def plot_error(num_coords, truth_table, est_state_mat, cov_bounds, AC_dt, C):
    # Make timestep for plot
    timestep = np.zeros(num_coords)
    t = 0
    for i in range(len(timestep)):
        t += AC_dt[i]
        timestep[i] = t

    # Truth table should match timestep length
    truth_table = truth_table[:num_coords,:]

    # Make covariance upper and lower bound
    # Sigma bounds for plot
    cov_sigma = 2
    up_bound = cov_sigma * cov_bounds
    lw_bound = cov_sigma * cov_bounds*-1

    # Make error between truth and predicted state
    # Remove acceleration 
    est_state_mat = np.delete(est_state_mat, 2, 1)
    est_state_mat = np.delete(est_state_mat, 4, 1)
    est_state_mat = np.delete(est_state_mat, 6, 1)

    # Find error between states
    state_error = truth_table - est_state_mat

    # Plotting Truth vs Predicted User Coords x-axis
    plt.figure()
    plt.title('User coordinates error for x-axis (ENU)')
    plt.plot(timestep, state_error[:,0], label = "Error")
    plt.plot(timestep, up_bound[:,0], color = 'black', label = "Upper Bound")
    plt.plot(timestep, lw_bound[:,0], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords Error x-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User Coords y-axis
    plt.figure()
    plt.title('User coordinates error for y-axis (ENU)')
    plt.plot(timestep, state_error[:,2], label = "Error")
    plt.plot(timestep, up_bound[:,2], color = 'black', label = "Upper Bound")
    plt.plot(timestep, lw_bound[:,2], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords Error y-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User Coords z-axis
    plt.figure()
    plt.title('User coordinates error for z-axis (ENU)')
    plt.plot(timestep, state_error[:,4], label = "Error")
    plt.plot(timestep, up_bound[:,4], color = 'black', label = "Upper Bound")
    plt.plot(timestep, lw_bound[:,4], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Coords Error z-axis (m)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity x-axis
    plt.figure()
    plt.title('User Velocity error for x-axis (ENU)')
    plt.plot(timestep, state_error[:,1], label = "Error")
    plt.plot(timestep, up_bound[:,1], color = 'black', label = "Upper Bound")
    plt.plot(timestep, lw_bound[:,1], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity Error x-axis (m/s)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity y-axis
    plt.figure()
    plt.title('User Velocity error for y-axis (ENU)')
    plt.plot(timestep, state_error[:,3], label = "Error")
    plt.plot(timestep, up_bound[:,3], color = 'black', label = "Upper Bound")
    plt.plot(timestep, lw_bound[:,3], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity Error y-axis (m/s)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity z-axis
    plt.figure()
    plt.title('User Velocity error for z-axis (ENU)')
    plt.plot(timestep, state_error[:,5], label = "Error")
    plt.plot(timestep, up_bound[:,5], color = 'black', label = "Upper Bound")
    plt.plot(timestep, lw_bound[:,5], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Velocity Error z-axis (m/s)')
    plt.legend()

    # Plotting Truth vs Predicted User velocity z-axis
    plt.figure()
    plt.title('User Clock error')
    plt.plot(timestep, state_error[:,6]/C, label = "Error")
    plt.plot(timestep, up_bound[:,6]/C, color = 'black', label = "Upper Bound")
    plt.plot(timestep, lw_bound[:,6]/C, color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Clock Error')
    plt.legend()

    
    plt.show()
    return    

def plot_res(SV_res_mat, thres_mat, cum_res_mat, AC_dt):
    # Make timestep for plot
    timestep = np.zeros(len(AC_dt))
    t = 0
    for i in range(len(timestep)):
        res_arr = SV_res_mat[i]
        t += AC_dt[i]
        timestep[i] = t

    # Set zero values to nan to plot dropout 
    SV_res_mat = np.where(SV_res_mat != 0, SV_res_mat, np.nan)


    #Plotting SV0 (PRN 3) residual and threshold each timestep
    plt.figure()
    plt.title('SV0 residual vs Time')
    plt.plot(timestep, SV_res_mat[:, 0], label = "SV0 Residual")
    plt.plot(timestep, thres_mat[:], label = "Threshold")
    plt.xlabel('Time (secs)')
    plt.ylabel('Residual (m)')
    plt.legend()

    #Plotting SV1 (PRN 4) residual and threshold each timestep
    plt.figure()
    plt.title('SV1 residual vs Time')
    plt.plot(timestep, SV_res_mat[:, 1], label = "SV1 Residual")
    plt.plot(timestep, thres_mat[:], label = "Threshold")
    plt.xlabel('Time (secs)')
    plt.ylabel('Residual (m)')
    plt.legend()

    #Plotting SV2 (PRN 9) residual and threshold each timestep
    plt.figure()
    plt.title('SV2 residual vs Time')
    plt.plot(timestep, SV_res_mat[:, 2], label = "SV2 Residual")
    plt.plot(timestep, thres_mat[:], label = "Threshold")
    plt.xlabel('Time (secs)')
    plt.ylabel('Residual (m)')
    plt.legend()

    #Plotting SV3 (PRN 16) residual and threshold each timestep
    plt.figure()
    plt.title('SV3 residual vs Time')
    plt.plot(timestep, SV_res_mat[:, 3], label = "SV3 Residual")
    plt.plot(timestep, thres_mat[:], label = "Threshold")
    plt.xlabel('Time (secs)')
    plt.ylabel('Residual (m)')
    plt.legend()

    #Plotting SV4 (PRN 22) residual and threshold each timestep
    plt.figure()
    plt.title('SV4 residual vs Time')
    plt.plot(timestep, SV_res_mat[:, 4], label = "SV4 Residual")
    plt.plot(timestep, thres_mat[:], label = "Threshold")
    plt.xlabel('Time (secs)')
    plt.ylabel('Residual (m)')
    plt.legend()

    #Plotting SV5 (PRN 26) residual and threshold each timestep
    plt.figure()
    plt.title('SV5 residual vs Time')
    plt.plot(timestep, SV_res_mat[:, 5], label = "SV5 Residual")
    plt.plot(timestep, thres_mat[:], label = "Threshold")
    plt.xlabel('Time (secs)')
    plt.ylabel('Residual (m)')
    plt.legend()

    #Plotting SV6 (PRN 27) residual and threshold each timestep
    plt.figure()
    plt.title('SV6 residual vs Time')
    plt.plot(timestep, SV_res_mat[:, 6], label = "SV6 Residual")
    plt.plot(timestep, thres_mat[:], label = "Threshold")
    plt.xlabel('Time (secs)')
    plt.ylabel('Residual (m)')
    plt.legend()

    #Plotting SV7 (PRN 31) residual and threshold each timestep
    plt.figure()
    plt.title('SV7 residual vs Time')
    plt.plot(timestep, SV_res_mat[:, 7], label = "SV7 Residual")
    plt.plot(timestep, thres_mat[:], label = "Threshold")
    plt.xlabel('Time (secs)')
    plt.ylabel('Residual (m)')
    plt.legend()

    #Plotting residual and threshold each timestep
    plt.figure()
    plt.title('Cumulative Residual vs Time')
    plt.plot(timestep, cum_res_mat[:], label = "Cum Residual")
    plt.plot(timestep, thres_mat[:], label = "Threshold")
    plt.xlabel('Time (secs)')
    plt.ylabel('Cumulative Residual (m)')
    plt.legend()

    plt.show()

    return

## SET UP
# ENU Center fixed point (Geodetic Coords in radians)
ENU_cfp = np.array([0.686675684196376, -1.501148870448501, 187.0243855202571])
# ENU Center fixed point (ECEF m) matlab conversion
ENU_cfp_ECEF = np.array([343736.907462, -4927400.792919, 4022010.073744])
# satellite timestep (update rate)
s_dt = 1
# speed of light (m/s)
C = 299792458.0
# Psuedorange bias
Cdt = -0.000410830353*C
# Pseudorange std dev
Pr_std = 0.0

# Real AC data
GPS_PR, GPS_PR_0, GPS_pos_0, GPS_pos_matrix, AC_dt, AC_x0, truth_table = gen_flight_data(ENU_cfp, ENU_cfp_ECEF, Cdt)

# Insert initial PR and GPS position
# GPS_pos_matrix = np.insert(GPS_pos_matrix, 0, GPS_pos_0, axis=0)
# GPS_PR = np.insert(GPS_PR, 0, GPS_PR_0, axis=0)

# Add in accceleration for initial state
acc0 = np.array([0.,0.,0.])
AC_x0 = np.insert(AC_x0, (2,4,6), (acc0[0], acc0[1], acc0[2]))
# number of coordinates/steps from user not including initial
num_coords = int(len(AC_dt))
# num_coords = 10
# number of satellites
num_SVs = 8
# Initial Covariance (don't know bias)
P0 = np.zeros((len(AC_x0),len(AC_x0)))
d_array = [100,50,25,100,50,25,100,50,25,100,100]
np.fill_diagonal(P0, d_array)


# Set current state and current covariance
curr_x = AC_x0
curr_P = P0 

# Build User Estimated States Matrix
est_state_mat = np.zeros((num_coords, len(AC_x0)))

# Build Estimated Covariance Matirx
est_cov_mat = np.zeros((num_coords, len(AC_x0), len(AC_x0)))

# Standard covariance for plotting bounds
cov_bounds = np.zeros((num_coords, len(AC_x0)))

# To store the sensor measurement and changes for plotting
sens_meas_mat_plot = np.zeros((num_coords, num_SVs))
sens_meas_mat = []

# Residual matrix
res_mat = []
SV_res_mat = np.zeros((num_coords, num_SVs))
cum_res_mat = np.zeros((num_coords))
thres_mat = np.zeros((num_coords))

# Predicted Pseudorange Matrix
pred_mat = np.zeros((num_coords, num_SVs))
pred_meas_mat = []

## Make window size for residuals
res_win = []

for i in range(num_coords):
    # Pulling one per time step
    sens_meas = GPS_PR[i*8:i*8+8,:].flatten() 
    truth = truth_table[i]
    sat_ENU = GPS_pos_matrix[i*8:i*8+8,:]
    dt = AC_dt[i]

    # Check if there is any nan data
    nan_meas = np.isnan(sens_meas)
    idx_nan_meas = np.where(nan_meas == True)
    
    if nan_meas.all() == True:
        print(f'Timestep: {i}')
        print(f'ALL SATELLITE INFO IS UNNAVAILABLE')
        print('\n')

        # EKF
        curr_x, curr_P, K, H, res, wtd_norm_res, pred_meas  = EKF(sat_ENU, sens_meas, dt, curr_x, curr_P)
        SV_res_mat[i] = np.insert(res, idx_nan_meas[0], 0)

    elif nan_meas.any() == True and nan_meas.all() == False:
        print(f'Timestep: {i}')
        print(f'Satellite {idx_nan_meas} info is unnavailable')
        print('\n')
        sens_meas = np.delete(sens_meas, idx_nan_meas)
        sat_ENU = np.delete(sat_ENU, idx_nan_meas, axis=0)

        # EKF
        curr_x, curr_P, K, H, res, wtd_norm_res, pred_meas  = EKF(sat_ENU, sens_meas, dt, curr_x, curr_P)

        # RAIM chi2 global statistic check
        res_win.append(wtd_norm_res)
        res_win, cum_res, thres = RAIM_chi2_global(res, res_win)

        # # Check if test statistic is within chi squared model for cumulative residual
        # if cum_res < thres:
        #     print(f'Coordinate Point {i} is valid')
        #     print(f'All SVs are valid')
        #     print('\n')
            
        # else:
        #     spoofed_sat = np.argmax(np.absolute(res))
        #     spoofed_sat_res = np.max(np.absolute(res))
        #     print(f'Coordinate Point {i} is invalid')
        #     print(f'Satellites Residuals: {res}')
        #     print(f'Satellite {spoofed_sat} issue')
        #     print(f'{spoofed_sat_res} m off')
        #     print('\n')

        #     # Timestep when bias was caught
        #     bias_catch_ts = i
        #     # RAIM chi2 sequential local statistic check
        #     H, res, K, sat_ENU, sens_meas, pred_meas = local_seq_test(curr_x, sens_meas, rho_error, i, sat_ENU, spoofed_sat)
        #     # Clear residual window
        #     res_win.clear()
        #     # Add zero to end of sensor measurement and predicted measurement for plotting
        #     sens_meas = np.append(sens_meas, 0.0)
        #     pred_meas = np.append(pred_meas, 0.0)

        # Update state and covariance
        curr_x = curr_x + K.dot(res)
        curr_P = curr_P - K.dot(H).dot(curr_P)

        # Standardize Covariance for plot bounds
        std_cov = np.sqrt(curr_P.diagonal())
        
        # # Store for plotting
        res_mat.append(res)

        # Add in zero where naan data was for plotting nan dropout in plot
        nan_arr = idx_nan_meas[0]
        for cnt in range(len(nan_arr)):
            res = np.insert(res, nan_arr[cnt], 0)
        SV_res_mat[i] = res

        sens_meas_mat.append(sens_meas)
        pred_meas_mat.append(pred_meas)
        # sens_meas_mat_plot[i] = sens_meas
        # cum_res_mat[i] = cum_res
        # thres_mat[i] = thres
        est_state_mat[i] = curr_x
        est_cov_mat[i] = curr_P
        cov_bounds[i] = std_cov
        # pred_mat[i] = pred_meas
    
    else:
        # EKF
        curr_x, curr_P, K, H, res, wtd_norm_res, pred_meas  = EKF(sat_ENU, sens_meas, dt, curr_x, curr_P)

        # RAIM chi2 global statistic check
        res_win.append(wtd_norm_res)
        res_win, cum_res, thres = RAIM_chi2_global(res, res_win)

        # # Check if test statistic is within chi squared model for cumulative residual
        # if cum_res < thres:
        #     print(f'Coordinate Point {i} is valid')
        #     print(f'All SVs are valid')
        #     print('\n')
            
        # else:
        #     spoofed_sat = np.argmax(np.absolute(res))
        #     spoofed_sat_res = np.max(np.absolute(res))
        #     print(f'Coordinate Point {i} is invalid')
        #     print(f'Satellites Residuals: {res}')
        #     print(f'Satellite {spoofed_sat} issue')
        #     print(f'{spoofed_sat_res} m off')
        #     print('\n')

        #     # Timestep when bias was caught
        #     bias_catch_ts = i
        #     # RAIM chi2 sequential local statistic check
        #     H, res, K, sat_ENU, sens_meas, pred_meas = local_seq_test(curr_x, sens_meas, rho_error, i, sat_ENU, spoofed_sat)
        #     # Clear residual window
        #     res_win.clear()
        #     # Add zero to end of sensor measurement and predicted measurement for plotting
        #     sens_meas = np.append(sens_meas, 0.0)
        #     pred_meas = np.append(pred_meas, 0.0)

        # Update state and covariance
        curr_x = curr_x + K.dot(res)
        curr_P = curr_P - K.dot(H).dot(curr_P)

        # Standardize Covariance for plot bounds
        std_cov = np.sqrt(curr_P.diagonal())
        
        # # Store for plotting
        res_mat.append(res)
        SV_res_mat[i] = res

        sens_meas_mat.append(sens_meas)
        pred_meas_mat.append(pred_meas)
        # sens_meas_mat_plot[i] = sens_meas
        # cum_res_mat[i] = cum_res
        # thres_mat[i] = thres
        est_state_mat[i] = curr_x
        est_cov_mat[i] = curr_P
        cov_bounds[i] = std_cov
        # pred_mat[i] = pred_meas

# Plotting and Tables
# Reorganize matrices to plot satellites accurately
# if spoofed_sat == 0:
#     valid_gps_0 = sens_meas_mat_plot[:bias_catch_ts,1]
#     valid_gps_0 = np.append(valid_gps_0, sens_meas_mat_plot[bias_catch_ts:,0])
#     valid_gps_1 = sens_meas_mat_plot[:bias_catch_ts,2]
#     valid_gps_1 = np.append(valid_gps_1, sens_meas_mat_plot[bias_catch_ts:,1])
#     valid_gps_2 = sens_meas_mat_plot[:bias_catch_ts,3]
#     valid_gps_2 = np.append(valid_gps_2, sens_meas_mat_plot[bias_catch_ts:,2])
#     valid_gps_3 = sens_meas_mat_plot[:bias_catch_ts,4]
#     valid_gps_3 = np.append(valid_gps_3, sens_meas_mat_plot[bias_catch_ts:,3])

#     pred_gps_0 = pred_mat[:bias_catch_ts,1]
#     pred_gps_0 = np.append(pred_gps_0, pred_mat[bias_catch_ts:,0])
#     pred_gps_1 = pred_mat[:bias_catch_ts,2]
#     pred_gps_1 = np.append(pred_gps_1, pred_mat[bias_catch_ts:,1])
#     pred_gps_2 = pred_mat[:bias_catch_ts,3]
#     pred_gps_2 = np.append(pred_gps_2, pred_mat[bias_catch_ts:,2])
#     pred_gps_3 = pred_mat[:bias_catch_ts,4]
#     pred_gps_3 = np.append(pred_gps_3, pred_mat[bias_catch_ts:,3])

# elif spoofed_sat == 1:
#     valid_gps_0 = sens_meas_mat_plot[:,0]
#     valid_gps_1 = sens_meas_mat_plot[:bias_catch_ts,2]
#     valid_gps_1 = np.append(valid_gps_1, sens_meas_mat_plot[bias_catch_ts:,1])
#     valid_gps_2 = sens_meas_mat_plot[:bias_catch_ts,3]
#     valid_gps_2 = np.append(valid_gps_2, sens_meas_mat_plot[bias_catch_ts:,2])
#     valid_gps_3 = sens_meas_mat_plot[:bias_catch_ts,4]
#     valid_gps_3 = np.append(valid_gps_3, sens_meas_mat_plot[bias_catch_ts:,3])

#     pred_gps_0 = pred_mat[:,0]
#     pred_gps_1 = pred_mat[:bias_catch_ts,2]
#     pred_gps_1 = np.append(pred_gps_1, pred_mat[bias_catch_ts:,1])
#     pred_gps_2 = pred_mat[:bias_catch_ts,3]
#     pred_gps_2 = np.append(pred_gps_2, pred_mat[bias_catch_ts:,2])
#     pred_gps_3 = pred_mat[:bias_catch_ts,4]
#     pred_gps_3 = np.append(pred_gps_3, pred_mat[bias_catch_ts:,3])

# elif spoofed_sat == 2:
#     valid_gps_0 = sens_meas_mat_plot[:,0]
#     valid_gps_1 = sens_meas_mat_plot[:,1]
#     valid_gps_2 = sens_meas_mat_plot[:bias_catch_ts,3]
#     valid_gps_2 = np.append(valid_gps_2, sens_meas_mat_plot[bias_catch_ts:,2])
#     valid_gps_3 = sens_meas_mat_plot[:bias_catch_ts,4]
#     valid_gps_3 = np.append(valid_gps_3, sens_meas_mat_plot[bias_catch_ts:,3])

#     pred_gps_0 = pred_mat[:,0]
#     pred_gps_1 = pred_mat[:,1]
#     pred_gps_2 = pred_mat[:bias_catch_ts,3]
#     pred_gps_2 = np.append(pred_gps_2, pred_mat[bias_catch_ts:,2])
#     pred_gps_3 = pred_mat[:bias_catch_ts,4]
#     pred_gps_3 = np.append(pred_gps_3, pred_mat[bias_catch_ts:,3])

# elif spoofed_sat == 3:
#     valid_gps_0 = sens_meas_mat_plot[:,0]
#     valid_gps_1 = sens_meas_mat_plot[:,1]
#     valid_gps_2 = sens_meas_mat_plot[:,2]
#     valid_gps_3 = sens_meas_mat_plot[:bias_catch_ts,4]
#     valid_gps_3 = np.append(valid_gps_3, sens_meas_mat_plot[bias_catch_ts:,3])

#     pred_gps_0 = pred_mat[:,0]
#     pred_gps_1 = pred_mat[:,1]
#     pred_gps_2 = pred_mat[:,2]
#     pred_gps_3 = pred_mat[:bias_catch_ts,4]
#     pred_gps_3 = np.append(pred_gps_3, pred_mat[bias_catch_ts:,3])

# elif spoofed_sat == 4:
#     valid_gps_0 = sens_meas_mat_plot[:,0]
#     valid_gps_1 = sens_meas_mat_plot[:,1]
#     valid_gps_2 = sens_meas_mat_plot[:,2]
#     valid_gps_3 = sens_meas_mat_plot[:,3]

#     pred_gps_0 = pred_mat[:,0]
#     pred_gps_1 = pred_mat[:,1]
#     pred_gps_2 = pred_mat[:,2]
#     pred_gps_3 = pred_mat[:,3]

# valid_sat_meas_mat = np.vstack((valid_gps_0,valid_gps_1,valid_gps_2,valid_gps_3)).T
# valid_pred_sat_meas_mat = np.vstack((pred_gps_0,pred_gps_1,pred_gps_2,pred_gps_3)).T
# bias_sat_meas_mat = np.append(bias_sat_meas_mat, sens_meas_mat_plot[bias_catch_ts:, -1])
# bias_sat_meas_pred_mat = np.append(bias_sat_meas_pred_mat, pred_mat[bias_catch_ts:, -1])


# Plot Psuedoranges of measurements and predicted measurements 
# plot_pseudo(valid_sat_meas_mat, bias_sat_meas_mat, valid_pred_sat_meas_mat, bias_sat_meas_pred_mat, num_coords, s_dt)
# Plot Truth coordinates to Predicted Coordinates
# plot_coords(num_coords, truth_table, est_state_mat, AC_dt)
# Plot Error with covariance bound
plot_error(num_coords, truth_table, est_state_mat, cov_bounds, AC_dt, C)
# Plot Cumulative Residual over time
plot_res(SV_res_mat, thres_mat, cum_res_mat, AC_dt)
print('done')
# Convert Residual data to CSV for Excel Table
# np.savetxt("residuals.csv", res_mat, delimiter=",")

