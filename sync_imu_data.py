import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def set_xvalues(polygon, x0):
    _ndarray = polygon.get_xy()
    _ndarray = np.array(_ndarray)
    new_coords = np.array([x0, _ndarray[1]])
    polygon.set_xy(new_coords)

def update(val):
    new_lag = lag_slider.val
    for i in range(0, len(stim_polygons)):
        poly = stim_polygons[i]
        onset = onsets[i]
        duration = durations[i]
        set_xvalues(poly, onset+new_lag)
    fig.canvas.draw_idle()

if __name__ == "__main__":

    # Where all the data is stored
    pq_folder = "imu_data_parquet"
    pq_files = [ f.path for f in os.scandir(pq_folder)]
    event_file = "temp_data/all_events_nirs.csv"

    # Read event data
    all_events = pd.read_csv(event_file)
    subjects = np.unique(all_events['subject'])
    sessions = ['protocol_1', 'protocol_2', 'protocol_3']

    # Some subjects need special handling
    subjects_from_raw_data = []

    fs_imu = 128
    result_event_file = "temp_data/all_events_nirs_imu.csv"
    for pq_file in pq_files:
        
        # Read IMU data
        imu_data = pd.read_parquet(pq_file)
        imu_subject = list(imu_data.columns)[0].split("/")[0]

        # Check if we've already processed this subject
        if os.path.exists(result_event_file):
            processed_events = pd.read_csv(result_event_file)
            processed_subjects = np.unique(processed_events['subject'])
            if imu_subject in processed_subjects:
                print("Already processed " + imu_subject + ", skipping")
                continue

        # Go through sessions
        new_events = []
        for session in sessions:
            
            # Get event data and acc data for session
            events = all_events[(all_events["subject"] == imu_subject) & (all_events["session"] == session.replace('_', ''))].copy()
            seek_column = session + "/LUMBAR/Accelerometer"
            acc_data = imu_data.filter(regex=seek_column).to_numpy()
            print("Using pq file " + pq_file)
            print(imu_data.head())
            
            # Some participants did not complete all protocols
            if acc_data.size == 0 or events.size == 0:
                print("Lacking IMU data for session " + session)
                continue

            # Start of IMU data should be at a constant offset from first trigger
            # This is about 7 seconds, countdown + start instruction
            first_onset = events['onset'].to_numpy()[0]
            events['onset'] = events['onset'] - first_onset + 7.0

            # Prepare a plot with a acc data
            fig, ax = plt.subplots(figsize=(14, 6))
            plt.subplots_adjust(left=0.25, bottom=0.25)
            xaxis = np.arange(0,acc_data.shape[0])
            xaxis_t = xaxis / fs_imu
            plt.plot(xaxis_t, acc_data[:,0])

            # Add stimulus blocks
            stim_polygons = []
            onsets = []
            durations = []
            for index, row in events.iterrows():
                onset = row['onset']
                duration = row['duration']
                lag_poly = plt.axvspan(onset, onset+duration, facecolor='lightseagreen', alpha=0.2, zorder=-100)
                stim_polygons.append(lag_poly)
                onsets.append(onset)
                durations.append(duration)

            # How much lag in slider?
            slider_min = -250
            slider_max = 200
            if imu_subject in subjects_from_raw_data:
                slider_min = -8000
                slider_max = 8000

            # Add a slider to adjust onset
            axcolor = 'lightgoldenrodyellow'
            slider_axis = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            lag_slider = Slider(slider_axis, 'Lag', slider_min, slider_max, valinit=0, valstep=0.1)
            lag_slider.on_changed(update)
            plt.show()  

            # Save adjusted onset and lag between signals
            lag = lag_slider.val
            for index, row in events.iterrows():
                events.at[index,'imu_fnirs_lag_seconds'] = lag
                events.at[index,'adjusted_onset_imu'] = events.at[index,'onset'] + lag
            new_events.append(events)
            print(events)

        # Concatenate data for subject and append to result csv
        all_events_new = pd.concat(new_events)
        all_events_new.to_csv("temp_data/all_events_nirs_imu.csv", index=False, mode="a")