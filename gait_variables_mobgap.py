import os
import sys
import pandas as pd
import numpy as np
import warnings

from mobgap.turning import TdElGohary
from mobgap.utils.conversions import to_body_frame
from mobgap.orientation_estimation import MadgwickAHRS

fs_imu = 128


def extract_condition_data(imu_data, session, sensor_position, onset, duration):

    # Get session data
    seek_column = f"{session}/{sensor_position}/Accelerometer"
    acc_data = imu_data.filter(regex=seek_column).to_numpy()
    seek_column = f"{session}/{sensor_position}/Gyroscope"
    gyr_data = imu_data.filter(regex=seek_column).to_numpy()
    seek_column = f"{session}/{sensor_position}/Magnetometer"
    mag_data = imu_data.filter(regex=seek_column).to_numpy()

    # Get condition data. Get the closest sample to onset.
    time_axis = np.arange(0,acc_data.shape[0]) / fs_imu
    onset_idx = (np.abs(time_axis - onset)).argmin()
    end_idx = (np.abs(time_axis - (onset+duration))).argmin()

    # Cut to condition block
    acc_data = acc_data[onset_idx:end_idx,:]
    gyr_data = gyr_data[onset_idx:end_idx,:]
    mag_data = mag_data[onset_idx:end_idx,:]

    # Convert magnetometer data from microT to milliT
    mag_data = mag_data / 1000

    # For lumbar: as opposed to gaitmap frame, keep the original Opal frame
    # but reverse axis. This aligns with mobgap example dataset acceleration directions.
    if sensor_position == "LUMBAR":
        acc_data = -acc_data
        gyr_data = -gyr_data
        mag_data = -mag_data

    # mobgap and gaitmap wants gyro in deg/s, not rad/s as we have
    gyr_data = gyr_data * 57.2957795

    return acc_data, gyr_data, mag_data


def construct_gaitmap_dataframe(acc_data, gyro_data, mag_data):
    time_axis = np.arange(0, acc_data.shape[0]) / fs_imu
    data = {
        "acc_x": acc_data[:,0],
        "acc_y": acc_data[:,1],
        "acc_z": acc_data[:,2],
        "gyr_x": gyro_data[:,0],
        "gyr_y": gyro_data[:,1],
        "gyr_z": gyro_data[:,2],
        "mag_x": mag_data[:,0],
        "mag_y": mag_data[:,1],
        "mag_z": mag_data[:,2]
    }
    df = pd.DataFrame(data, index=time_axis)
    return df

def construct_empty_param_dataframe():
    means_data = {
        'angle_deg': [],
        'duration_s': [],
    }
    means_frame = pd.DataFrame(data=means_data)
    return means_frame

def construct_param_dataframe(turn_list):

    # In case we have empty slices, suppress warnings from np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Get means and return parameters
        means_data = {
            'angle_deg': [turn_list['angle_deg'].abs().mean()],
            'duration_s': [turn_list['duration_s'].mean()],
        }
        means_frame = pd.DataFrame(data=means_data)
    
    return means_frame

def extract_variables(series_in, imu_data, session):
    trial_type = series_in['trial_type']
    imu_onset = float(series_in['adjusted_onset_imu'])
    trial_duration = float(series_in['duration'])
    subject = series_in['subject']
    protocol = series_in['session']

    # We are only interested in turning parameters from navigation blocks
    if trial_type not in ["Navigated_walking",
                          "Navigation",
                          "Navigation_and_Aud_Stroop"]:
        return series_in
    
    # FNP1068 had very strange behavior on gyro data in P1, where left foot cuts off halfway.
    if subject in ['FNP1068'] and protocol in ['protocol1']:
        return series_in
    
    print(f"Processing {subject} / {session} / onset {series_in['onset']}")

    # Prepare IMU data arrays for condition data
    acc_data_lf, gyro_data_lf, mag_data_lf = extract_condition_data(imu_data, session, "LEFT_FOOT", imu_onset, trial_duration)
    acc_data_rf, gyro_data_rf, mag_data_rf = extract_condition_data(imu_data, session, "RIGHT_FOOT", imu_onset, trial_duration)
    acc_data_lumbar, gyro_data_lumbar, mag_data_lumbar = extract_condition_data(imu_data, session, "LUMBAR", imu_onset, trial_duration)

    # construct dataframe lumbar
    lumbar_data = construct_gaitmap_dataframe(acc_data_lumbar, gyro_data_lumbar, mag_data_lumbar)

    # give it a samples column and re-index
    lumbar_data['samples'] = range(len(lumbar_data))
    lumbar_data.index = range(len(lumbar_data))

    # reorient to body frame
    lumbar_data = to_body_frame(lumbar_data)

    # turn detection
    # these have been optimized to correspond
    # to Mobility Lab output when comparing 
    # all blocks at the summary level
    try:
        orientation_estimator = MadgwickAHRS()
        turning_detector = TdElGohary(orientation_estimation=orientation_estimator,
                                      min_gap_between_turns_s=0.5,
                                      lower_threshold_velocity_dps=30)
        turning_detector.detect(lumbar_data, sampling_rate_hz=fs_imu)
        turn_list = turning_detector.turn_list_
        print(f"The following number of turns were identified: {len(turn_list)}")
    except:
        print("WARN: could not detect turns for block")
        return series_in
    
    # Make sure we found a turn
    if len(turn_list) >= 1:
        means_frame = construct_param_dataframe(turn_list)
    else:
        print("Not counting segment with no turns")
        means_frame = construct_empty_param_dataframe()

    # Create parameters for the whole block and add to passed list
    means_frame['trial_type'] = trial_type
    for column in means_frame.columns:
        if len(means_frame[column].values) > 1:
            print("WARN: unexpected amount of turns")
            sys.exit(1)
        if len(means_frame[column].values) == 1:
            series_in[column] = means_frame[column][0]
        else:
            series_in[column] = np.nan
    series_in["Turn Count"] = len(turn_list)

    return series_in


if __name__ == "__main__":

    # Where all the data is stored
    pq_folder = "../Park-MOVE_fnirs_dataset_v2/IMU_data/imu_data_parquet"
    pq_files = [ f.path for f in os.scandir(pq_folder)]

    # Read event data
    event_file = "../Park-MOVE_fnirs_dataset_v2/IMU_data/all_events_nirs_imu.csv"
    all_events = pd.read_csv(event_file)
    subjects = np.unique(all_events['subject'])
    sessions = ['protocol_1', 'protocol_2', 'protocol_3']

    # Go through each pq file
    calculated_param_frames = []
    for pq_file in pq_files:

        # Read IMU data
        imu_data = pd.read_parquet(pq_file)
        subject = list(imu_data.columns)[0].split("/")[0]

        # Skip subjects not included in study
        if subject in ['FNP1002','FNP1014', 'FNP1077']:
            print("Skipping " + subject)
            continue

        print("Using pq file " + pq_file)

        # Go through sessions
        new_events = []
        for session in sessions:
            print("On subject %s, session %s\n" % (subject, session))

            # Get event data and acc data for session
            events = all_events[(all_events["subject"] == subject) & (all_events["session"] == session.replace('_', ''))].copy()
            if events.empty:
                print(f"{subject} does not have {session}, skipping")
                continue

            # FNP1006 seems to have the label RIGHT_WRIST instead of RIGHT_FOOT on protocol 1. Replace.
            if subject == "FNP1006" and session == "protocol_1":
                old_columns = [column for column in imu_data if "RIGHT_WRIST" in column]
                new_columns = [column.replace("WRIST", "FOOT") for column in imu_data if "RIGHT_WRIST" in column]
                col_mapping = dict(zip(old_columns, new_columns))
                imu_data.rename(columns = col_mapping, inplace = True)
            
            # Go through each condition
            calculated_params = events.apply(extract_variables, axis=1, imu_data=imu_data, session=session)
            block_numbers = np.arange(start=1, stop=len(calculated_params['onset'])+1)
            calculated_params['block'] = block_numbers
            calculated_param_frames.append(calculated_params)

            # Print results for subject
            print(calculated_params)

    final_frame_gait = pd.concat(calculated_param_frames).drop(columns=["onset", "adjusted_onset_imu", "duration", "imu_fnirs_lag_seconds", "sample", "value"])
    for col in ['block','trial_type', 'session', 'subject']:
        final_frame_gait.insert(0, col, final_frame_gait.pop(col))
    print(final_frame_gait)
    final_frame_gait.to_csv("imu_turn_parameters.csv", index=False)

