import os
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt

from gaitmap.preprocessing import sensor_alignment
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap.event_detection import RamppEventDetection, FilteredRamppEventDetection, HerzerEventDetection
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory, MadgwickRtsKalman
from gaitmap.parameters import TemporalParameterCalculation
from gaitmap.parameters import SpatialParameterCalculation
from gaitmap.data_transform import ButterworthFilter

fs_imu = 128

def plot_gyro(dataset, threshold):
    signal = dataset["left_sensor"][["gyr_x", "gyr_y", "gyr_z"]].to_numpy()
    signal_norm = np.linalg.norm(signal, axis=1)
    _, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 5))
    ax0.set_title("Gyro for aligning against gravity!")
    ax0.set_ylabel("gyro deg/s")
    ax0.plot(signal)
    ax0.axhline(threshold, c="k", ls="--", lw=0.5)
    ax0.axhline(-threshold, c="k", ls="--", lw=0.5)
    ax1.set_title("Gyro norm")
    ax1.set_ylabel("gyro deg/s")
    ax1.plot(signal_norm)
    ax1.axhline(threshold, c="k", ls="--", lw=0.5)
    ax1.axhline(-threshold, c="k", ls="--", lw=0.5)
    plt.show()

def plot_stride_detection(dtw):
    sensor = "left_sensor"
    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(10, 5))
    dtw.data[sensor]["gyr_ml"].reset_index(drop=True).plot(ax=axs[0])
    axs[0].set_ylabel("gyro [deg/s]")
    axs[1].plot(dtw.cost_function_[sensor])
    axs[1].set_ylabel("dtw cost [a.u.]")
    axs[1].axhline(dtw.max_cost, color="k", linestyle="--")
    axs[2].imshow(dtw.acc_cost_mat_[sensor], aspect="auto")
    axs[2].set_ylabel("template position [#]")
    for p in dtw.paths_[sensor]:
        axs[2].plot(p.T[1], p.T[0])
    for s in dtw.matches_start_end_original_[sensor]:
        axs[1].axvspan(*s, alpha=0.3, color="g")
    for _, s in dtw.stride_list_[sensor][["start", "end"]].iterrows():
        axs[0].axvspan(*s, alpha=0.3, color="g")

    axs[0].set_xlabel("time [#]")
    fig.tight_layout()
    fig.show()
    plt.show()

def plot_event_detection(bf_data, ed):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 5))
    ax1.plot(bf_data["left_sensor"].reset_index(drop=True)[["gyr_ml"]])
    ax2.plot(bf_data["left_sensor"].reset_index(drop=True)[["acc_pa"]])

    ic_idx = ed.min_vel_event_list_["left_sensor"]["ic"].to_numpy().astype(int)
    tc_idx = ed.min_vel_event_list_["left_sensor"]["tc"].to_numpy().astype(int)
    min_vel_idx = ed.min_vel_event_list_["left_sensor"]["min_vel"].to_numpy().astype(int)

    for ax, sensor in zip([ax1, ax2], ["gyr_ml", "acc_pa"]):
        for _i, stride in ed.min_vel_event_list_["left_sensor"].iterrows():
            ax.axvline(stride["start"], color="g")
            ax.axvline(stride["end"], color="r")
        ax.scatter(
            ic_idx,
            bf_data["left_sensor"][sensor].to_numpy()[ic_idx],
            marker="*",
            s=100,
            color="r",
            zorder=3,
            label="ic",
        )
        ax.scatter(
            tc_idx,
            bf_data["left_sensor"][sensor].to_numpy()[tc_idx],
            marker="p",
            s=50,
            color="g",
            zorder=3,
            label="tc",
        )
        ax.scatter(
            min_vel_idx,
            bf_data["left_sensor"][sensor].to_numpy()[min_vel_idx],
            marker="s",
            s=50,
            color="y",
            zorder=3,
            label="min_vel",
        )
        ax.grid(True)

    ax1.set_title("Events of min_vel strides")
    ax1.set_ylabel("gyr_ml (°/s)")
    ax2.set_ylabel("acc_pa [m/s^2]")
    plt.legend(loc="best")

    fig.tight_layout()
    fig.show()
    plt.show()

def plot_trajectory_estimation(trajectory):
    try:
        # select the position of the first stride
        first_stride_position = trajectory.position_["left_sensor"].loc[0]
        first_stride_position.plot()
        plt.title("Left Foot Trajectory per axis")
        plt.xlabel("sample")
        plt.ylabel("position [m]")
        plt.show()
        # select the orientation of the first stride
        print(trajectory.orientation_["left_sensor"])
        first_stride_orientation = trajectory.orientation_["left_sensor"].loc[0]
        first_stride_orientation.plot()
        plt.title("Left Foot Orientation per axis")
        plt.xlabel("sample")
        plt.ylabel("orientation [a.u.]")
        plt.show()
    except:
        print("Could not plot trajectory")

# Class containing gait parameters for a bout of walking
class SegmentParameters:

    # Constructor
    def __init__(self, means_data, step_times_left, step_times_right, stride_lengths_left, stride_lengths_right):
        self.means_data = means_data
        self.step_times_left = step_times_left
        self.step_times_right = step_times_right
        self.stride_lengths_left = stride_lengths_left
        self.stride_lengths_right = stride_lengths_right
        self.trial_type = "None"

    # Factory function to initialize from multiple SegmentParameters
    # This will take the average of all segments for means data, and 
    # concatenate all vector data
    @classmethod
    def from_multiple_segments(cls, segments):

        # If we have no segments coming in, create empty object
        if len(segments) < 1:
            return cls(None, [], [], [], [])
        
        # Get the means
        segment_means = [segment.means_data for segment in segments]
        means_frame = pd.DataFrame([pd.concat(segment_means).mean(axis=0)])
        
        # Concatenate vector data
        step_times_left = np.concatenate([params.step_times_left for params in segments])
        step_times_right = np.concatenate([params.step_times_right for params in segments])
        stride_lengths_left = np.concatenate([params.stride_lengths_left for params in segments])
        stride_lengths_right = np.concatenate([params.stride_lengths_right for params in segments])

        # Create data class
        return cls(means_frame, step_times_left, step_times_right, stride_lengths_left, stride_lengths_right)
    
    # Copy from another SegmentParameters
    def copy(self, other):
        self.means_data = other.means_data
        self.step_times_left = other.step_times_left
        self.step_times_right = other.step_times_right
        self.stride_lengths_left = other.stride_lengths_left
        self.stride_lengths_right = other.stride_lengths_right

    # Set trial type
    def set_trial_type(self, trial_type):
        self.trial_type = trial_type

# Calculate variability parameters according to Galna et al 2013
def calculate_variability_parameters(block_parameters, subject, session):

    # Group by trial type
    trial_types = np.unique([params.trial_type for params in block_parameters])
    all_frames = []
    for trial in trial_types:
        trial_block_parameters = [params for params in block_parameters if params.trial_type == trial]
        trial_parameters = SegmentParameters.from_multiple_segments(trial_block_parameters)
        
        # Variability calculation
        step_time_sd_lr = np.sqrt(
            (np.var(trial_parameters.step_times_left) + 
            np.var(trial_parameters.step_times_right)) / 2)
        stride_length_sd_lr = np.sqrt(
            (np.var(trial_parameters.stride_lengths_left) + 
            np.var(trial_parameters.stride_lengths_right)) / 2)
        
        # For asymmetry: take difference of each gait cycle
        # e.g. step_times_left[0] - step_times_right[0]
        # Asymmetry step time
        step_time_min_length = np.min([len(trial_parameters.step_times_left), len(trial_parameters.step_times_right)])
        lr_array = np.array([np.array(trial_parameters.step_times_left[0:step_time_min_length]).astype(float),
                             np.array(trial_parameters.step_times_right[0:step_time_min_length]).astype(float)]).astype(float)
        mean_array = np.mean(lr_array, axis=0)
        step_time_diff_array = np.abs(np.diff(lr_array, axis=0))
        step_time_diff_array_percent = np.divide(step_time_diff_array, mean_array)*100

        # Asymmetry stride length
        stride_length_min_length = np.min([len(trial_parameters.stride_lengths_left), len(trial_parameters.stride_lengths_right)])
        lr_array = np.array([np.array(trial_parameters.stride_lengths_left[0:stride_length_min_length]).astype(float),
                             np.array(trial_parameters.stride_lengths_right[0:stride_length_min_length]).astype(float)]).astype(float)
        mean_array = np.mean(lr_array, axis=0)
        stride_length_diff_array = np.abs(np.diff(lr_array, axis=0))
        stride_length_diff_array_percent = np.divide(stride_length_diff_array, mean_array)*100

        # Compile parameters
        var_data = {
            'subject': [subject],
            'session': [session],
            'trial_type': [trial],
            'Step Time Variability': [step_time_sd_lr],
            'Step Time Variability Amount Used Values (L+R)': [len(trial_parameters.step_times_left) + len(trial_parameters.step_times_right)],
            'Stride Length Variability': [stride_length_sd_lr],
            'Stride Length Variability Amount Used Values (L+R)': [len(trial_parameters.stride_lengths_left) + len(trial_parameters.stride_lengths_right)],
            'Step Time Asymmetry': [np.mean(step_time_diff_array)],
            'Step Time Asymmetry Percent': [np.mean(step_time_diff_array_percent)],
            'Step Time Asymmetry Amount Used Values': [step_time_min_length],
            'Stride Length Asymmetry': [np.mean(stride_length_diff_array)],
            'Stride Length Asymmetry Percent': [np.mean(stride_length_diff_array_percent)],
            'Stride Length Asymmetry Amount Used Values': [stride_length_min_length],
        }
        var_frame = pd.DataFrame(data=var_data)
        all_frames.append(var_frame)
    
    session_var_frame = pd.concat(all_frames)
    return session_var_frame

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

    # For lumbar: Z should be X and reverse Y and X
    if sensor_position == "LUMBAR":
        acc_data = np.transpose(np.array([-acc_data[:,2], -acc_data[:,1], -acc_data[:,0]])) 
        gyr_data = np.transpose(np.array([-gyr_data[:,2], -gyr_data[:,1], -gyr_data[:,0]])) 
        mag_data = np.transpose(np.array([-mag_data[:,2], -mag_data[:,1], -mag_data[:,0]])) 

    # gaitmap wants gyro in deg/s, not rad/s as we have
    gyr_data = gyr_data * 57.2957795

    return acc_data, gyr_data, mag_data

def get_segment_data(acc_data, gyr_data, mag_data, segment_start, segment_end):
    acc_data = acc_data[segment_start:segment_end,:]
    gyr_data = gyr_data[segment_start:segment_end,:]
    mag_data = mag_data[segment_start:segment_end,:]
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

def get_step_times(ed, temporal_paras):
    stride_idx_left = ed.min_vel_event_list_['left_sensor'].index.to_numpy()
    stride_idx_right = ed.min_vel_event_list_['right_sensor'].index.to_numpy()
    common_strides = np.intersect1d(stride_idx_left, stride_idx_right)
    step_times_left = []
    step_times_right = []
    for stride_idx in common_strides:
        # They count stride from foot flat to foot flat (min vel)
        hs_left = ed.min_vel_event_list_['left_sensor']['ic'][stride_idx]
        hs_right = ed.min_vel_event_list_['right_sensor']['ic'][stride_idx]
        to_left = ed.min_vel_event_list_['left_sensor']['tc'][stride_idx]
        to_right = ed.min_vel_event_list_['right_sensor']['tc'][stride_idx]
        stance_left = temporal_paras.parameters_['left_sensor']['stance_time'][stride_idx]
        stance_right = temporal_paras.parameters_['right_sensor']['stance_time'][stride_idx]
        # On which side does the combined gait cycle start?
        if to_left < to_right:
            step_time_right = (hs_right - hs_left) / fs_imu
            step_time_left = stance_right - (to_right - hs_left) / fs_imu
        else:
            step_time_left = (hs_left - hs_right) / fs_imu
            step_time_right = stance_left - (to_left - hs_right) / fs_imu
        # Sometimes this can result in a negative value 
        # where the strides are not actually part of the same cycle.
        # Skip these.
        if step_time_left < 0 or step_time_right < 0:
            continue
        step_times_left.append(step_time_left)
        step_times_right.append(step_time_right)

    return step_times_left, step_times_right

def construct_empty_param_dataframe():
    means_data = {
        'Cadence L': [],
        'Single Support L': [],
        'Step Time L': [],
        'Stride Length L': [],
        'Walking Speed L': [],
        'Swing Time L': [],
        'Stance Time L': [],
        'Stride Time L': [],
        'Cadence R': [],
        'Single Support R': [],
        'Step Time R': [],
        'Stride Length R': [],
        'Walking Speed R': [],
        'Stance Time R': [],
        'Stride Time R': [],
    }
    means_frame = pd.DataFrame(data=means_data)
    segment_data = SegmentParameters(means_frame, [], [], [], [])
    return means_frame, segment_data

def get_cadence(step_times):
    cadence = np.divide(np.ones(step_times.shape), step_times) * 60
    return cadence

def construct_param_dataframe(ed, temporal_paras, spatial_paras):
    step_times_lf, step_times_rf = get_step_times(ed, temporal_paras)
    step_times_lf = np.array(step_times_lf)
    step_times_rf = np.array(step_times_rf)
    cadence_lf = get_cadence(step_times_lf)
    cadence_rf = get_cadence(step_times_rf)
    tss_times_lf = temporal_paras.parameters_['right_sensor']['swing_time'].to_numpy()
    tss_times_rf = temporal_paras.parameters_['left_sensor']['swing_time'].to_numpy()
    stride_lengths_lf = spatial_paras.parameters_['left_sensor']['stride_length'].to_numpy()
    stride_lengths_rf = spatial_paras.parameters_['right_sensor']['stride_length'].to_numpy()
    walking_speeds_lf = spatial_paras.parameters_['left_sensor']['gait_velocity'].to_numpy()
    walking_speeds_rf = spatial_paras.parameters_['right_sensor']['gait_velocity'].to_numpy()
    stance_lf = temporal_paras.parameters_['left_sensor']['stance_time'].to_numpy()
    stance_rf = temporal_paras.parameters_['right_sensor']['stance_time'].to_numpy()
    stride_lf = temporal_paras.parameters_['left_sensor']['stride_time'].to_numpy()
    stride_rf = temporal_paras.parameters_['right_sensor']['stride_time'].to_numpy()

    # In case we have empty slices, suppress warnings from np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Get means and return parameters
        means_data = {
            'Cadence L': [np.mean(cadence_lf)],
            'Single Support L': [np.mean(tss_times_lf)],
            'Step Time L': [np.mean(step_times_lf)],
            'Stride Length L': [np.mean(stride_lengths_lf)],
            'Walking Speed L': [np.mean(walking_speeds_lf)],
            'Stance Time L': [np.mean(stance_lf)],
            'Stride Time L': [np.mean(stride_lf)],
            'Cadence R': [np.mean(cadence_rf)],
            'Single Support R': [np.mean(tss_times_rf)],
            'Step Time R': [np.mean(step_times_rf)],
            'Stride Length R': [np.mean(stride_lengths_rf)],
            'Walking Speed R': [np.mean(walking_speeds_rf)],
            'Stance Time R': [np.mean(stance_rf)],
            'Stride Time R': [np.mean(stride_rf)],
        }
        means_frame = pd.DataFrame(data=means_data)
    segment_data = SegmentParameters(means_frame, step_times_lf, step_times_rf, stride_lengths_lf, stride_lengths_rf)
    
    return means_frame, segment_data

def construct_param_dataframe_slice(ed, temporal_paras, spatial_paras, start, end):
    step_times_lf, step_times_rf = get_step_times(ed, temporal_paras)
    step_times_lf = np.array(step_times_lf[start:end])
    step_times_rf = np.array(step_times_rf[start:end])
    cadence_lf = get_cadence(step_times_lf)
    cadence_rf = get_cadence(step_times_rf)
    tss_times_lf = temporal_paras.parameters_['right_sensor']['swing_time'].to_numpy()[start:end]
    tss_times_rf = temporal_paras.parameters_['left_sensor']['swing_time'].to_numpy()[start:end]
    stride_lengths_lf = spatial_paras.parameters_['left_sensor']['stride_length'].to_numpy()[start:end]
    stride_lengths_rf = spatial_paras.parameters_['right_sensor']['stride_length'].to_numpy()[start:end]
    walking_speeds_lf = spatial_paras.parameters_['left_sensor']['gait_velocity'].to_numpy()[start:end]
    walking_speeds_rf = spatial_paras.parameters_['right_sensor']['gait_velocity'].to_numpy()[start:end]
    stance_lf = temporal_paras.parameters_['left_sensor']['stance_time'].to_numpy()[start:end]
    stance_rf = temporal_paras.parameters_['right_sensor']['stance_time'].to_numpy()[start:end]
    stride_lf = temporal_paras.parameters_['left_sensor']['stride_time'].to_numpy()[start:end]
    stride_rf = temporal_paras.parameters_['right_sensor']['stride_time'].to_numpy()[start:end]

    # In case we have empty slices, suppress warnings from np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Get means and return parameters
        means_data = {
            'Cadence L': [np.mean(cadence_lf)],
            'Single Support L': [np.mean(tss_times_lf)],
            'Step Time L': [np.mean(step_times_lf)],
            'Stride Length L': [np.mean(stride_lengths_lf)],
            'Walking Speed L': [np.mean(walking_speeds_lf)],
            'Stance Time L': [np.mean(stance_lf)],
            'Stride Time L': [np.mean(stride_lf)],
            'Cadence R': [np.mean(cadence_rf)],
            'Single Support R': [np.mean(tss_times_rf)],
            'Step Time R': [np.mean(step_times_rf)],
            'Stride Length R': [np.mean(stride_lengths_rf)],
            'Walking Speed R': [np.mean(walking_speeds_rf)],
            'Stance Time R': [np.mean(stance_rf)],
            'Stride Time R': [np.mean(stride_rf)],
        }
        means_frame = pd.DataFrame(data=means_data)
    segment_data = SegmentParameters(means_frame, step_times_lf, step_times_rf, stride_lengths_lf, stride_lengths_rf)
    
    return means_frame, segment_data

def extract_variables(series_in, imu_data, session, block_parameters, calibration_data, plot_gait_events=False):
    trial_type = series_in['trial_type']
    imu_onset = float(series_in['adjusted_onset_imu'])
    trial_duration = float(series_in['duration'])
    subject = series_in['subject']
    protocol = series_in['session']

    # Standing audio stroop has no gait parameters
    if trial_type == "Stand_still_and_Aud_Stroop":
        return series_in
    
    # FNP1068 had very strange behavior on gyro data in P1, where left foot cuts off halfway.
    if subject in ['FNP1068'] and protocol in ['protocol1']:
        return series_in
    
    print(f"Processing {subject} / {session} / onset {series_in['onset']}")

    # Prepare IMU data arrays for condition data
    acc_data_lf, gyro_data_lf, mag_data_lf = extract_condition_data(imu_data, session, "LEFT_FOOT", imu_onset, trial_duration)
    acc_data_rf, gyro_data_rf, mag_data_rf = extract_condition_data(imu_data, session, "RIGHT_FOOT", imu_onset, trial_duration)
    acc_data_lumbar, gyro_data_lumbar, mag_data_lumbar = extract_condition_data(imu_data, session, "LUMBAR", imu_onset, trial_duration)

    # Which direction are we facing?
    segment_mag_avg = np.mean(mag_data_lumbar[0:100,1])
    direction = 'forward'
    if segment_mag_avg < 0:
        direction = 'backward'

    # Approximate default orientation if no calibration data
    # This is in (x,y,z,w)
    if direction == 'forward':
        q0_default = [-0.320, 0.149, 0.900, 0.256]
    else:
        q0_default = [0.149, -0.320, -0.900, 0.256]

    # Set initial orientation
    calibration_data_dir = calibration_data[calibration_data['direction'] == direction]
    if len(calibration_data_dir) < 1:
        q0_lf = q0_default
        q0_rf = q0_default
        print(f"Using default q0 {q0_default}")
    else:
        calibration_data_dir_lf = calibration_data_dir[calibration_data_dir['sensor'] == "LEFT_FOOT"]
        calibration_data_dir_rf = calibration_data_dir[calibration_data_dir['sensor'] == "RIGHT_FOOT"]
        q0_lf = [calibration_data_dir_lf['q0_x'].values[0], calibration_data_dir_lf['q0_y'].values[0],
                 calibration_data_dir_lf['q0_z'].values[0], calibration_data_dir_lf['q0_w'].values[0]]
        q0_rf = [calibration_data_dir_rf['q0_x'].values[0], calibration_data_dir_rf['q0_y'].values[0],
                 calibration_data_dir_rf['q0_z'].values[0], calibration_data_dir_rf['q0_w'].values[0]]
        print(f"Using q0 LF {q0_lf}")
        print(f"Using q0 RF {q0_rf}")
    
    # construct dataframe left/right
    lf_data = construct_gaitmap_dataframe(acc_data_lf, gyro_data_lf, mag_data_lf)
    rf_data = construct_gaitmap_dataframe(acc_data_rf, gyro_data_rf, mag_data_rf)
    gm_dataset = {"left_sensor": lf_data, "right_sensor": rf_data}

    # First align to gravity, and then put in body frame
    threshold = 60
    if plot_gait_events:
        plot_gyro(gm_dataset, threshold)
    
    # This step can fail; catch if so.
    try:
        dataset_sf_aligned_to_gravity = sensor_alignment.align_dataset_to_gravity(gm_dataset, fs_imu, 
                                                                                static_signal_th=threshold, 
                                                                                window_length_s=0.1)
        bf_data = convert_to_fbf(dataset_sf_aligned_to_gravity, left_like="left_", right_like="right_")
    except:
        print("WARN: failed to align to gravity, returning null parameters")
        return series_in
    
    # Segmentation parameters optimized for trial type
    # These are default unless we have navigation blocks
    dtw_cost = 4.0
    dtw_min_len = 0.6
    ed = RamppEventDetection()
    if trial_type in ['Navigated_walking', 'Navigation', 'Navigation_and_Aud_Stroop']:
        dtw_cost = 5.0
        dtw_min_len = 0.4
        ed = RamppEventDetection()

    # Segment data into strides
    dtw = BarthDtw(max_cost=dtw_cost, min_match_length_s=dtw_min_len)
    dtw = dtw.segment(data=bf_data, sampling_rate_hz=fs_imu)
    if plot_gait_events:
        plot_stride_detection(dtw)

    # Find events
    ed = ed.detect(data=bf_data, stride_list=dtw.stride_list_, sampling_rate_hz=fs_imu)
    if plot_gait_events:
        plot_event_detection(bf_data, ed)

    traj_method = MadgwickRtsKalman(initial_orientation=q0_lf)
    trajectory = StrideLevelTrajectory(trajectory_method=traj_method, ori_method=None, pos_method=None)
    trajectory = trajectory.estimate(
        data=dataset_sf_aligned_to_gravity, stride_event_list=ed.min_vel_event_list_, sampling_rate_hz=fs_imu
    )
    if plot_gait_events:
        plot_trajectory_estimation(trajectory)
    
    temporal_paras = TemporalParameterCalculation()
    temporal_paras = temporal_paras.calculate(stride_event_list=ed.min_vel_event_list_, sampling_rate_hz=fs_imu)
    spatial_paras = SpatialParameterCalculation()
    spatial_paras = spatial_paras.calculate(
        stride_event_list=ed.min_vel_event_list_,
        positions=trajectory.position_,
        orientations=trajectory.orientation_,
        sampling_rate_hz=fs_imu,
    )

    print(
        f"The following number of strides were identified and parameterized for each sensor: {({k: len(v) for k, v in ed.min_vel_event_list_.items()})}"
    )

    no_strides_left = len(ed.min_vel_event_list_['left_sensor'])
    no_strides_right = len(ed.min_vel_event_list_['right_sensor'])

    # If we have enough strides in straight walking, skip the first and last two strides to avoid acc/deacc
    if trial_type in ["Straight_walking", "Straight_walking_and_Aud_Stroop"]:
        if no_strides_left >= 7 and no_strides_right >= 7:
            means_frame, segment_data = construct_param_dataframe_slice(ed, temporal_paras, spatial_paras, 2, -3)
        else:
            print("Not counting segment with few strides")
            means_frame, segment_data = construct_empty_param_dataframe()
    else:
        # Skip just the first strides
        means_frame, segment_data = construct_param_dataframe_slice(ed, temporal_paras, spatial_paras, 1, -1)

    # Did we get a session with 0 detected strides?
    if no_strides_left == 0 or no_strides_right == 0:
        return series_in

    # Create parameters for the whole block and add to passed list
    segment_data.set_trial_type(trial_type)
    block_parameters.append(segment_data)

    # Add values to series passed into the function and return
    for column in segment_data.means_data:
        if len(segment_data.means_data[column].values) >= 1:
            series_in[column] = segment_data.means_data[column].values[0]
        else:
            series_in[column] = np.nan
    series_in["Step Count L"] = len(segment_data.step_times_left)
    series_in["Step Count R"] = len(segment_data.step_times_right)
    return series_in


if __name__ == "__main__":

    plot_gait_events = 0
    plot_filter = 0
    plot_quaternions_on = 0
    plot_positions = 0
    plot_trajectory = 0

    # Where all the data is stored
    pq_folder = "../Park-MOVE_fnirs_dataset_v2/IMU_data/imu_data_parquet"
    pq_files = [ f.path for f in os.scandir(pq_folder)]

    # Read calibration data
    calibration_file = "../Park-MOVE_fnirs_dataset_v2/IMU_data/calibration_stance_data.csv"
    calibration_data = pd.read_csv(calibration_file)

    # Read event data
    event_file = "../Park-MOVE_fnirs_dataset_v2/IMU_data/all_events_nirs_imu.csv"
    all_events = pd.read_csv(event_file)
    subjects = np.unique(all_events['subject'])
    sessions = ['protocol_1', 'protocol_2', 'protocol_3']

    # Read subject height data
    height_file = "../Park-MOVE_fnirs_dataset_v2/basic_demographics.csv"
    subject_height_data = pd.read_csv(height_file)

    # Prepare folder for figures
    output_folder_name = os.getcwd() + "/saved_figures"
    if not (os.path.isdir(output_folder_name)):
        os.mkdir(output_folder_name)
    
    # Go through each pq file
    calculated_param_frames = []
    variability_frames = []
    for pq_file in pq_files:

        # Read IMU data
        imu_data = pd.read_parquet(pq_file)
        subject = list(imu_data.columns)[0].split("/")[0]

        # Skip subjects not included in study
        if subject in ['FNP1002','FNP1014', 'FNP1077']:
            print("Skipping " + subject)
            continue

        # Get subject height for use in pendulum model
        subject_height = subject_height_data[subject_height_data['subject'] == subject]['height'].values[0]
        if np.isnan(subject_height):
            print("WARN: no height data, setting to None")
            subject_height = None

        # Get calibration data
        subject_calibration_data = calibration_data[calibration_data['subject'] == subject]

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
            
            # Prepare a list for holding detailed block values
            block_parameters = []

            # Go through each condition
            calculated_params = events.apply(extract_variables, axis=1, imu_data=imu_data, session=session, 
                                             plot_gait_events=plot_gait_events, block_parameters=block_parameters,
                                             calibration_data=subject_calibration_data)
            block_numbers = np.arange(start=1, stop=len(calculated_params['onset'])+1)
            calculated_params['block'] = block_numbers
            calculated_param_frames.append(calculated_params)

            # Print results for subject
            print(calculated_params)

            # With data from all blocks, also calculate variability data
            if len(block_parameters) > 1:
                variability_data = calculate_variability_parameters(block_parameters, subject, session.replace('_', ''))
                variability_frames.append(variability_data)
                print(variability_data)

    final_frame_gait = pd.concat(calculated_param_frames).drop(columns=["onset", "adjusted_onset_imu", "duration", "imu_fnirs_lag_seconds", "sample", "value"])
    for col in ['block','trial_type', 'session', 'subject']:
        final_frame_gait.insert(0, col, final_frame_gait.pop(col))
    print(final_frame_gait)
    final_frame_gait.to_csv("imu_gait_parameters.csv", index=False)

    final_frame_variability = pd.concat(variability_frames)
    print(final_frame_variability)
    final_frame_variability.to_csv("imu_variability_parameters.csv", index=False)
    print('done')