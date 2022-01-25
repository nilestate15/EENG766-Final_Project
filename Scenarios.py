import numpy as np
import pymap3d as pm



def scenario_1(num_coords, Cdt, Cdt_dot):
    '''
    This function is used to generate simulated data of satellite location, receiver state, and pseudorange measurements
    to be used in "RBRAIM_flightdata_Model2.py" (meaning the structure of the data will be made to be run in this file)
    Stationary satellite data pulled in on 22 Jan 2022 19:44:07 (EST)

    Args:
    num_SVs: a integer representing the number of satellites to be used when creating data
    origin: a (3,) array of the starting position of the receiver

    Returns: 
    PR: A (# of timesteps, # of satellites) array of the psuedoranges from each satellite for each timestep (array excludes initial pseudoranges).
    PR0: A (# of Satellites, 1) array of the initial pseudorange of each satellite.
    GPS_pos0: A (# of Satellites, 3) array of the initial position of each satellite.
    GPS_pos_mat: A (# of Satellites * # timestep of timesteps, 3) array of each satellite position for each timestep (array excludes initial satellite positions).
    dt_arr: A (# of timesteps,) array of the difference in time between each data point timestamp.
    AC_x0: A (# of states,) array of the initial state of the aircraft receiver.
    truth_table: A (# of timesteps, # of states) array of the state of the aircraft for each timestep. (array exclude initial aircraft state) 
    '''
    # Difference in time between data
    dt = 1.0
    # Matrix of dts for each coordinate
    dt_arr = np.ones(num_coords)*dt
    # Measurement noise
    rho_error = 2.0

    # Center fixed point Geodetic (degrees) Beavercreek, OH
    geo_cfp = np.array([39.71, -84.06, 267])

    # Stationary Satellite position data in Geodetic (degrees)
    GPS1 = np.array([55.8, -21.6, 19941000])
    GPS3 = np.array([16.7, -27.8, 20108000])
    GPS6 = np.array([-3.4, -101.7, 20145000])
    GPS12 = np.array([21.3, 159.8, 20028000])
    GPS13 = np.array([-3.4, -149.6, 20284000])
    GPS15 = np.array([9.6, -174.0, 20291000])
    GPS17 = np.array([52.0, -96.7, 20532000])
    GPS19 = np.array([38.5, -121.1, 20095000])

    # Put GPS data into a single array
    GPS_pos_geo = np.array([GPS1,GPS3,GPS6,GPS12,GPS13,GPS15,GPS17,GPS19]).reshape((8,3))

    # Convert geodetic coordinates to ENU
    GPS_pos0 = np.zeros((len(GPS_pos_geo),3))
    for i in range(len(GPS_pos_geo)):
        gps = GPS_pos_geo[i,:]
        GPS_pos0[i,:] = pm.geodetic2enu(gps[0],gps[1],gps[2],geo_cfp[0],geo_cfp[1],geo_cfp[2])

    enu_cfp = pm.geodetic2enu(geo_cfp[0],geo_cfp[1],geo_cfp[2],geo_cfp[0],geo_cfp[1],geo_cfp[2])
    enu_cfp = np.array(enu_cfp)

    ## Make GPS position matrix 
    # Velocity of satellites to be moving to make varying satellite positions (flight data satellite velocity avg for GPS3 E (136.09) N (-2624.57) U (-1820.23))
    rand_vel = np.random.default_rng()
    GPS_pos_mat = np.zeros((len(GPS_pos0)*num_coords, 3))
    curr_GPS = GPS_pos0
    for k in range(num_coords): 
        curr_vel_E = rand_vel.normal(136.09, 0.5, len(GPS_pos0))
        curr_vel_N = rand_vel.normal(-110.57, 0.5, len(GPS_pos0))
        curr_vel_U = rand_vel.normal(-120.23, 0.5, len(GPS_pos0))
        curr_GPS_vel = np.vstack((curr_vel_E,curr_vel_N,curr_vel_U)).T
        curr_GPS = curr_GPS + curr_GPS_vel
        GPS_pos_mat[k*len(GPS_pos0):k*len(GPS_pos0)+len(GPS_pos0),:] = curr_GPS
        

    ## Make State Matrix
    # Make varying velocity for state (m/s)
    AC_vel0 = np.array([0,0,0])
    AC_vel_E = rand_vel.normal(2.0, 0.5, num_coords)
    AC_vel_N = rand_vel.normal(-3.5, 0.5, num_coords)
    AC_vel_U = rand_vel.normal(1.5, 0.5, num_coords)
    AC_vel = np.vstack((AC_vel_E,AC_vel_N,AC_vel_U)).T

    # Make AC position matrix 
    AC_pos0 = enu_cfp
    AC_pos_mat = np.zeros((num_coords, 3))
    curr_AC = AC_pos0
    for n in range(num_coords):
        curr_AC_vel = AC_vel[n,:]
        curr_AC = curr_AC + curr_AC_vel
        AC_pos_mat[n,:] = curr_AC
        

    # Initial State
    AC_x0 = [AC_pos0[0], AC_vel0[0], AC_pos0[1], AC_vel0[1], AC_pos0[2], AC_vel0[2], Cdt, Cdt_dot]

    # Make state matrix
    truth_table = np.zeros((len(AC_pos_mat), 8))
    for i in range(len(truth_table)):
        truth_table[i, :] = [AC_pos_mat[i,0], AC_vel[i,0], AC_pos_mat[i,1], AC_vel[i,1], AC_pos_mat[i,2], AC_vel[i,2], Cdt, Cdt_dot]

    # Make Pseudorange Measurements
    # Predicted Pseudorange Measurement (h(x) formula)
    PR = np.zeros((num_coords,8))
    # White noise for sensor pseudorange measurements
    white_noise = np.random.default_rng()

    PR0 = np.zeros(8)
    for int, sat_pos0 in enumerate(GPS_pos0):
        PR0[int] = np.sqrt((sat_pos0[0] - AC_x0[0])**2 + (sat_pos0[1] - AC_x0[2])**2 + (sat_pos0[2] - AC_x0[4])**2) + Cdt + white_noise.normal(0.0, 2, 1)

    for l in range(num_coords):
        GPS_pos = GPS_pos_mat[l*8:l*8+8,:]
        truth = truth_table[l]
        for m, sat_pos in enumerate(GPS_pos):
            PR[l,m] = np.sqrt((sat_pos[0] - truth[0])**2 + (sat_pos[1] - truth[2])**2 + (sat_pos[2] - truth[4])**2) + Cdt + white_noise.normal(0.0, 2, 1)

    return PR, PR0, GPS_pos0, GPS_pos_mat, dt_arr, AC_x0, truth_table 

def scenario_1_bias(PR):
    '''
    This function takes in the simulated data from scenario 1 and adds a bias to two satellites.
    Set to adding bias on satellite 0 at 200 secs and satellite 2 at 400 secs.

    Args:
    PR: A (# of timesteps, # of satellites) array of the psuedoranges before bias from each satellite for each timestep (array excludes initial pseudoranges).

    Returns:
    PR: A (# of timesteps, # of satellites) array of the psuedoranges after bias from each satellite for each timestep (array excludes initial pseudoranges).
    '''

    # How many satellites to be biased
    num_sats = 2
    # Add random bias to satellite for anomaly (m)
    sat_bias = 70
    # How many secs/meas you want to add random bias to
    num_bias = 10
    # List of which satellite to bias
    bias_sat_arr = np.array([0, 1])
    # List of when to bias satellite
    bias_sec_arr = np.array([200, 400])

    for i in range(num_sats):
        bias_sat = bias_sat_arr[i]
        bias_sec = bias_sec_arr[i]
        PR[bias_sec:bias_sec+num_bias,bias_sat] = PR[bias_sec:bias_sec+num_bias,bias_sat] + sat_bias

        # Print where the anomally will be
        print(f'Satellite w/ anomally: {bias_sat}')
        print(f'Where the anomally starts in timestep: {bias_sec}')

    return PR

def scenario_2(PR):
    '''
    This function takes in the the flight data from RBRAIM_flightdata_Model2 simultaneously jams two satellites.
    Set to jamming on satellites 1 and 5 at 700 secs.

    Args:
    PR: A (# of timesteps, # of satellites) array of the psuedoranges before jamming from each satellite for each timestep (array excludes initial pseudoranges).

    Returns:
    PR: A (# of timesteps, # of satellites) array of the psuedoranges after jamming from each satellite for each timestep (array excludes initial pseudoranges).
    '''
    # Reshape Array
    PR = PR.reshape(int(len(PR)/8),8)
    # How many satellites to be jammed
    num_sats = 2
    # List of which satellite to be jammed
    jam_sat_arr = np.array([1, 5])
    # List of when to jam satellite
    jam_sec_arr = np.array([700, 700])

    for i in range(num_sats):
        jam_sat = jam_sat_arr[i]
        jam_sec = jam_sec_arr[i]
        PR[jam_sec:,jam_sat] = np.nan

        # Print where the anomally will be
        print(f'Satellite jammed: {jam_sat}')
        print(f'Where the jamming starts in timestep: {jam_sec}')

    return PR

def scenario_3(PR):
    '''
    This function takes in the the flight data from RBRAIM_flightdata_Model2 simultaneously jams 4 satellites.
    Set to jamming on satellites 0, 2, 4, and 6 starting at 200 and jamming the others at a 200s interval.

    Args:
    PR: A (# of timesteps, # of satellites) array of the psuedoranges before jamming from each satellite for each timestep (array excludes initial pseudoranges).

    Returns:
    PR: A (# of timesteps, # of satellites) array of the psuedoranges after jamming from each satellite for each timestep (array excludes initial pseudoranges).
    '''
    # Reshape Array
    PR = PR.reshape(int(len(PR)/8),8)
    # How many satellites to be jammed
    num_sats = 4
    # List of which satellite to be jammed
    jam_sat_arr = np.array([0, 2, 4, 6])
    # List of when to jam satellite
    jam_sec_arr = np.array([200, 400, 800, 1000])

    for i in range(num_sats):
        jam_sat = jam_sat_arr[i]
        jam_sec = jam_sec_arr[i]
        PR[jam_sec:,jam_sat] = np.nan

        # Print where the anomally will be
        print(f'Satellite jammed: {jam_sat}')
        print(f'Where the jamming starts in timestep: {jam_sec}')

    return PR

def scenario_4(PR):
    '''
    This function takes in the the flight data from RBRAIM_flightdata_Model2 simultaneously biasing two satellites.
    Set to biasing satellites 2 and 6 at 800 secs.

    Args:
    PR: A (# of timesteps, # of satellites) array of the psuedoranges before bias from each satellite for each timestep (array excludes initial pseudoranges).

    Returns:
    PR: A (# of timesteps, # of satellites) array of the psuedoranges after bias from each satellite for each timestep (array excludes initial pseudoranges).
    '''
    # Reshape Array
    PR = PR.reshape(int(len(PR)/8),8)
    # How many satellites to be jammed
    num_sats = 2
    # Add random bias to satellite for anomaly (m)
    sat_bias = 70
    # How many secs/meas you want to add random bias to
    num_bias = 150
    # List of which satellite to be jammed
    bias_sat_arr = np.array([1, 5])
    # List of when to jam satellite
    bias_sec_arr = np.array([800, 800])

    for i in range(num_sats):
        bias_sat = bias_sat_arr[i]
        bias_sec = bias_sec_arr[i]
        PR[bias_sec:bias_sec+num_bias,bias_sat] = PR[bias_sec:bias_sec+num_bias,bias_sat] + sat_bias


        # Print where the anomally will be
        print(f'Satellite biased: {bias_sat}')
        print(f'Where the bias starts in timestep: {bias_sec}')

    return PR

