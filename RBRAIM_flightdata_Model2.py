from cgi import test
from multiprocessing.dummy import current_process
import numpy as np
import scipy.linalg as la
from math import nan, pi, sin, cos, sqrt
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from gnc import vanloan


def gen_flight_data(ENU_cfp, ENU_cfp_ECEF, Cdt, Cdt_dot):
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
    AC_x0 = np.array([AC_ENU[0,0], AC_vel[0,0], AC_ENU[0,1], AC_vel[0,1], AC_ENU[0,2], AC_vel[0,2], Cdt, Cdt_dot])
    # Remove initial position and velocity from array
    AC_ENU = np.delete(AC_ENU, 0, 0)
    AC_vel = np.delete(AC_vel, 0, 0)
    # Make truth matrix
    truth_table = np.zeros((len(AC_ENU), 8))
    for i in range(len(truth_table)):
        truth_table[i, :] = [AC_ENU[i,0], AC_vel[i,0], AC_ENU[i,1], AC_vel[i,1], AC_ENU[i,2], AC_vel[i,2], Cdt, Cdt_dot]
    
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


def EKF(sat_ENU, sens_meas, dt, curr_x, curr_P, Cdt):
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

    # Pseudorange measurement error equal variance
    # Needs to be std dev
    rho_error = 2.0

    # Build State Error Covariance Matrix
    Q = np.zeros((len(curr_x),len(curr_x)))
    Q_acc = 2.0*(accelbias_sigma**2)/accelbias_tau
    Q__cd = 2.0*(clockbias_sigma**2)/clockbias_tau
    Q_diag = [0, 0, Q_acc, 0, 0, Q_acc, 0, 0, Q_acc, 0, Q__cd]
    np.fill_diagonal(Q, Q_diag)
    # Scaling Q to be a smaller value for tuning
    Q *= 0.00305

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
    F_pva = np.array([[0.,1.,0.], 
                    [0.,0.,1.],
                    [0.,0.,(-1/accelbias_tau)]])


    # Influence of clock bias and clock drift on state
    # F_cb = np.array([[(-1/clockbias_tau),0], 
    #                 [0.,0]])

    F_cb = np.array([[0,1.],
                    [0,(-1/clockbias_tau)]])

    for m in range(int(len(F)/3)):
        F[3*m:3*m+3,3*m:3*m+3] = F_pva
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
        pred_meas[n] = np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[3])**2 + (sat_pos[2] - curr_x[6])**2) + curr_x[9]

    
    # Residuals (eq 26/eq 32 but using hx formula rather than Hx)
    res = sens_meas - pred_meas

    # Residual Covariance Matrix
    res_cov = H.dot(curr_P).dot(H.T) + R
    
    # Kalman Gain
    K = (curr_P.dot(H.T)).dot(la.inv(res_cov))


    # # Take Cholesky of residual covariance
    # # Equation given by Dr.Leishman from matlab code
    # L_res_cov = np.linalg.cholesky(res_cov)

    # ## Normalize Residual
    # # Equation given by Dr.Leishman from matlab code
    # norm_res = la.inv(L_res_cov).dot(res)

    # Standardize residuals for local test
    st_res = np.zeros(len(res))
    # Pull out diagonal of res covariance matrix
    res_cov_diag = np.diagonal(res_cov)
    for i in range(len(res)):
        st_res[i] = np.abs(res[i]/np.sqrt(res_cov_diag[i]))
    
    # Sum of Square Residual
    SS_res = res.T.dot(res)

    return curr_x, curr_P, K, H, res, res_cov, st_res, SS_res, pred_meas



def RAIM_chi2_global(res):
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

    ## Test statistic (RAIM Portion)
    # GPS RAIM: Statistics Based Improvement on the Calculation of Threshold and Horizontal Protection Radius Jiaxing Liu, Mingquan Lu, Zhenming Feng
    test_stat = np.sqrt(res.T.dot(res))
    # GNSS Signal Reliability Testing in Urban and Indoor Environments Heidi Kuusniemi*, Gérard Lachapelle
    # Test Statistic Variance
    # test_stat_var = res.T.dot(res)/(len(res)-4)
    # In aviation systems Pfa is fixed to Pfa is fixed at 1/15000.
    global_Pfa = 1/15000

    local_Pfa = .0001
    b0 = .19
   
    # Method of finding ab equal noncentrality parameter for both global and local test. (allows relation between global test alpha and local test alpha)
    # GNSS Signal Reliability Testing in Urban and Indoor Environments Heidi Kuusniemi*, Gérard Lachapelle
    n1alp = st.norm.ppf(q=1-(local_Pfa/2))
    n1bet = st.norm.ppf(q=1-b0)
    local_thres = (n1alp + n1bet)
    thres_b = (st.chi2.ppf(q = b0, df= 4,loc=local_thres**2))
    thres_a = st.chi2.ppf(q = 1-global_Pfa, df= 4)
    
    
    # GNSS Signal Reliability Testing in Urban and Indoor Environments Heidi Kuusniemi*, Gérard Lachapelle
    # Threshold variance
    # thres_var = st.chi2.ppf(q = 1-global_Pfa, df= (len(res)-4))/(len(res)-4)

    # GPS RAIM: Statistics Based Improvement on the Calculation of Threshold and Horizontal Protection Radius Jiaxing Liu, Mingquan Lu, Zhenming Feng
    # Find inverse chi squared for threshold (normalized chi2 threshold w/ num of satellites - 4 degrees of freedom [pos and clock bias])
    thres = st.chi2.ppf(q = 1-global_Pfa, df= 4)
        
    return test_stat, thres, local_Pfa

def local_seq_test(res, res_cov, st_res, st_res_err, num_st_res_err, n, curr_x, curr_P, sens_meas, sat_ENU):
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
        
    Returns:
        nothing
    '''
    if n != 0:
        # Standardize residuals for local test
        st_res = np.zeros(len(res))
        # Pull out diagonal of res covariance matrix
        res_cov_diag = np.diagonal(res_cov)
        for i in range(len(res)):
            st_res[i] = np.abs(res[i]/np.sqrt(res_cov_diag[i]))

        for k in range(len(st_res)):
            if st_res[k] < local_thres:
                print(f'Standardized Res {k} ACCEPTABLE')
        
            else:
                print(f'Standardized Res {k} ERRONEOUS')
                st_res_err.append(st_res[k])
    
        st_res_err = np.array(st_res_err)
    
    st_res_max = np.max(st_res_err)
    st_res_max_idx = np.where(st_res == st_res_max)
    sat_ENU = np.delete(sat_ENU, st_res_max_idx)
    sens_meas = np.delete(sens_meas, st_res_max_idx)

    # Build Measurement Error Covariance Matrix
    rho_error = 2.0
    R = np.eye(len(sens_meas)) * rho_error**2

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
        pred_meas[n] = np.sqrt((sat_pos[0] - curr_x[0])**2 + (sat_pos[1] - curr_x[3])**2 + (sat_pos[2] - curr_x[6])**2) + curr_x[9]

    # Residuals (eq 26/eq 32 but using hx formula rather than Hx)
    res = sens_meas - pred_meas

    # Residual Covariance Matrix
    res_cov = H.dot(curr_P).dot(H.T) + R
    
    # Kalman Gain
    K = (curr_P.dot(H.T)).dot(la.inv(res_cov))

    # Sum of Square Residual
    SS_res = res.T.dot(res)

    # Subtract of recursive local test tracker
    num_st_res_err =- 1
        
    return sens_meas, sat_ENU, H, pred_meas, res, K, SS_res, num_st_res_err


def plot_error(num_coords, truth_table, est_state_mat, cov_bounds, AC_dt):
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

    # Plotting Truth vs Predicted User Clock Error
    plt.figure()
    plt.title('User Clock error')
    plt.plot(timestep, state_error[:,6], label = "Error")
    plt.plot(timestep, up_bound[:,6], color = 'black', label = "Upper Bound")
    plt.plot(timestep, lw_bound[:,6], color = 'black', label = "Lower Bound")
    plt.xlabel('Time (secs)')
    plt.ylabel('User Clock Error')
    plt.legend()

    
    plt.show()
    return state_error

def plot_NEES(num_coords, state_error, NEES_cov_mat, AC_dt):
    # Make timestep for plot
    timestep = np.zeros(num_coords)
    t = 0
    for i in range(len(timestep)):
        t += AC_dt[i]
        timestep[i] = t 

    ## Normalized Estimation Error Squared (NEES)
    # Remove Clock bias and drift from state error
    # NEES_st_err = np.delete(state_error, [6,7], 1)
    # Remove velocity Clock bias and drift from state error
    NEES_st_err = np.delete(state_error, [1,3,5,6,7], 1)

    # NEES Matrix
    NEES_mat = np.zeros(num_coords)
    for i in range(num_coords):
        NEES_mat[i] = NEES_st_err[i].T.dot(la.inv(NEES_cov_mat[i])).dot(NEES_st_err[i])

    # Line of stability for filter
    stab_line = np.ones(num_coords)*3

    mean_NEES = np.mean(NEES_mat)

    # Plotting Truth vs Predicted User Coords x-axis
    plt.figure()
    plt.title('NEES of Filter')
    plt.plot(timestep, NEES_mat, label = "NEES")
    plt.plot(timestep, stab_line, label = "Stability")
    plt.xlabel('Time (secs)')
    plt.ylabel('NEES')
    plt.legend()
   
    plt.show()
    return          

def plot_res(SV_res_mat, thres_mat, AC_dt):
    # Make timestep for plot
    timestep = np.zeros(len(AC_dt))
    t = 0
    for i in range(len(timestep)):
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
# Psuedorange clock bias
Cdt = -0.000410830353*C
# Clock Drift
Cdt_dot = 0.0
# Pseudorange std dev
Pr_std = 0.0

# Real AC data
GPS_PR, GPS_PR_0, GPS_pos_0, GPS_pos_matrix, AC_dt, AC_x0, truth_table = gen_flight_data(ENU_cfp, ENU_cfp_ECEF, Cdt, Cdt_dot)

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
d_array = [100,1,10000,100,1,10000,100,1,10000,1,64000000]
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
sens_meas_mat = []

# Residual matrix
res_mat = []
SV_res_mat = np.zeros((num_coords, num_SVs))
thres_mat = np.zeros((num_coords))

# Predicted Pseudorange Matrix
pred_mat = np.zeros((num_coords, num_SVs))
pred_meas_mat = []

# Normalized Estimation Error Squared (NEES) Matrix
# NEES_cov_mat = np.zeros((num_coords, 6, 6))
NEES_cov_mat = np.zeros((num_coords, 3, 3))
# mean_res = np.zeros((len(num_coords)))

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
        print(f'All Satellite info is unnavailable')
        print('\n')

        # EKF
        curr_x, curr_P, K, H, res, res_cov, st_res, SS_res, pred_meas  = EKF(sat_ENU, sens_meas, dt, curr_x, curr_P, Cdt)
        SV_res_mat[i] = np.insert(res, idx_nan_meas[0], 0)

    elif nan_meas.any() == True and nan_meas.all() == False:
        print(f'Timestep: {i}')
        print(f'Satellite {idx_nan_meas} info is unnavailable')
        print('\n')
        sens_meas = np.delete(sens_meas, idx_nan_meas)
        sat_ENU = np.delete(sat_ENU, idx_nan_meas, axis=0)

        # EKF
        curr_x, curr_P, K, H, res, res_cov, st_res, SS_res, pred_meas  = EKF(sat_ENU, sens_meas, dt, curr_x, curr_P, Cdt)

        # RAIM chi2 global statistic check
        test_stat, thres, local_Pfa = RAIM_chi2_global(res)

        # Check if test statistic is within chi squared model for test statistic
        if test_stat < thres:
            print(f'GLOBAL TEST SUCCESS')
            print('\n')
            
        else:
            print(f'GLOBAL TEST FAILURE')
            print(f'Coordinate Point {i} Integrity Failure')
            print('\n')

            print(f'LOCAL TEST:')
            # Timestep when bias was caught
            bias_catch_ts = i

            local_thres = st.norm.ppf(q=1-(local_Pfa/2))
            st_res_err = []
            num_st_res_err = 0

            for k in range(len(st_res)):
                if st_res[k] < local_thres:
                    print(f'Standardized Res {k} ACCEPTABLE')
            
                else:
                    print(f'Standardized Res {k} ERRONEOUS')
                    st_res_err.append(st_res[k])
                    num_st_res_err += 1

            st_res_err = np.array(st_res_err)
            is_empty = st_res_err.size == 0

            if is_empty == False:
                print(f'RECURSIVE LOCAL TEST:')
                for n in range(num_st_res_err):
                    sens_meas, sat_ENU, H, pred_meas, res, K, SS_res, num_st_res_err = local_seq_test(res, res_cov, st_res, st_res_err, num_st_res_err, n, curr_x, curr_P, sens_meas, sat_ENU)

            # Rerun Global Test for final verification
            test_stat, thres, local_Pfa = RAIM_chi2_global(res)

            # Check if test statistic is within chi squared model for test statistic
            print('FINAL GLOBAL TEST:')
            if test_stat < thres:
                print(f'Position Estimate {i} ACCEPTABLE')
                print('\n')
                
            else:
                print(f'Position Estimate {i} NOT RELIABLE')
                print('\n')

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
        est_state_mat[i] = curr_x
        est_cov_mat[i] = curr_P
        cov_bounds[i] = std_cov

        # Saving Position and Velocity Covariance
        NEES_cov = curr_P
        NEES_cov = np.delete(NEES_cov, [2,5,8,9,10], 1)
        NEES_cov = np.delete(NEES_cov, [2,5,8,9,10], 0)

        # Remove velocity from covariance 
        NEES_cov = np.delete(NEES_cov, [1,3,5], 1)
        NEES_cov = np.delete(NEES_cov, [1,3,5], 0)
        NEES_cov_mat[i] = NEES_cov

    
    else:
        # EKF
        curr_x, curr_P, K, H, res, res_cov, st_res, SS_res, pred_meas  = EKF(sat_ENU, sens_meas, dt, curr_x, curr_P, Cdt)

        # RAIM chi2 global statistic check
        test_stat, thres, local_Pfa = RAIM_chi2_global(res)

        # Check if test statistic is within chi squared model for test statistic
        if test_stat < thres:
            print(f'GLOBAL TEST SUCCESS')
            print('\n')
            
        else:
            print(f'GLOBAL TEST FAILURE')
            print(f'Coordinate Point {i} Integrity Failure')
            print('\n')

            print(f'LOCAL TEST:')
            # Timestep when bias was caught
            bias_catch_ts = i

            local_thres = st.norm.ppf(q=1-(local_Pfa/2))
            st_res_err = []
            num_st_res_err = 0

            for k in range(len(st_res)):
                if st_res[k] < local_thres:
                    print(f'Standardized Res {k} ACCEPTABLE')
            
                else:
                    print(f'Standardized Res {k} ERRONEOUS')
                    st_res_err.append(st_res[k])
                    num_st_res_err += 1

            st_res_err = np.array(st_res_err)
            is_empty = st_res_err.size == 0

            if is_empty == False:
                print(f'RECURSIVE LOCAL TEST:')
                for n in range(num_st_res_err):
                    sens_meas, sat_ENU, H, pred_meas, res, K, SS_res, num_st_res_err = local_seq_test(res, res_cov, st_res, st_res_err, num_st_res_err, n, curr_x, curr_P, sens_meas, sat_ENU)

            # Rerun Global Test for final verification
            test_stat, thres, local_Pfa = RAIM_chi2_global(res)

            # Check if test statistic is within chi squared model for test statistic
            print('FINAL GLOBAL TEST:')
            if test_stat < thres:
                print(f'Position Estimate {i} ACCEPTABLE')
                print('\n')
                
            else:
                print(f'Position Estimate {i} NOT RELIABLE')
                print('\n')

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
        est_state_mat[i] = curr_x
        est_cov_mat[i] = curr_P
        cov_bounds[i] = std_cov

        # Saving Position and Velocity Covariance
        NEES_cov = curr_P
        NEES_cov = np.delete(NEES_cov, [2,5,8,9,10], 1)
        NEES_cov = np.delete(NEES_cov, [2,5,8,9,10], 0)
        # Remove velocity from covariance 
        NEES_cov = np.delete(NEES_cov, [1,3,5], 1)
        NEES_cov = np.delete(NEES_cov, [1,3,5], 0)
        NEES_cov_mat[i] = NEES_cov


# Plot Error with covariance bound
# state_error = plot_error(num_coords, truth_table, est_state_mat, cov_bounds, AC_dt)
# Plot Normalized Estimation Error Squared (NEES)
# plot_NEES(num_coords, state_error, NEES_cov_mat, AC_dt)
# Plot Cumulative Residual over time
plot_res(SV_res_mat, thres_mat, AC_dt)
print('done')
# Convert Residual data to CSV for Excel Table
# np.savetxt("residuals.csv", res_mat, delimiter=",")

