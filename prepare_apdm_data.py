import os 
import h5py
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import matplotlib.pyplot as plt

# Extract sensor data
def extract_sensor_into_frame(subject_id, file_protocol, sensor, label, types, axes):

    # Extract data from each sensor and store column-wise
    frame_columns = []
    frame_column_names = []
    for data_type in types:
        for axis in axes:
            sensor_data = f['Sensors'][sensor][data_type]
            axis_number = axes.index(axis)
            sensor_data_axis = pd.DataFrame(sensor_data[:,axis_number])
            column_name = "%s/%s/%s/%s/%s" % (subject_id, file_protocol, label, data_type, axis)
            frame_columns.append(sensor_data_axis)
            frame_column_names.append(column_name)

    # Concatenate sensor data into a dataframe and store metadata as column names
    sensor_frame = pd.concat(frame_columns, axis=1)
    sensor_frame.columns = frame_column_names
    return sensor_frame

# We're going to go through the HDF5 files for each subject, extract relevant data
# and store it in parquet format (column-wise time series, data compressed per column, metadata about columns, highly efficient column read)
if __name__ == "__main__":
        
    # Where all the data is stored
    root_data_folder = "P:/uMove/fNIRS/fNIRS_big_study/Data/APDM"
    subfolders = [ f.path for f in os.scandir(root_data_folder) if f.is_dir() ]

    # We know we're missing some calibration files
    missing_calibration = ["FNP1001", "FNP1005", "FNP1006", "FNP1007"]

    # These ones have the static ground calibration last instead of first
    calibration_last = ["FNP2023", "FNP2025", "FNP2037"]

    # Set up a mapping from file to protocol
    file_id_mapping_normal = {1: "calibration_1", 2: "calibration_2", 3: "protocol_1", 4: "protocol_2", 5: "protocol_3"}
    file_id_mapping_missing = {1: "protocol_1", 2: "protocol_2", 3: "protocol_3"}
    file_id_mapping_last = {1: "calibration_2", 2: "protocol_1", 3: "protocol_2", 4: "protocol_3", 5: "calibration_1", }
    file_id_mapping_1033 = {1: "calibration_1", 2: "calibration_2", 3: "protocol_2", 4: "protocol_1", 5: "protocol_3"}
    file_id_mapping_2040 = {1: "calibration_1", 2: "calibration_2", 3: "protocol_1", 4: "calibration_1_2", 5: "calibration_2_2", 6: "protocol_2"}

    for subfolder in subfolders:

        # Find the raw data folder
        raw_data_folder = [ f.path for f in os.scandir(subfolder) if "rawData" in f.path]
        if len(raw_data_folder) > 1:
            print("ERROR: more than one rawData folder for " + subfolder)
            exit()

        # Process the raw data files
        raw_data_folder = raw_data_folder[0]
        raw_data_files = [ f.path for f in os.scandir(raw_data_folder)]
        subject_id = raw_data_folder.split('/')[-1].split('\\')[1].split("_")[4]
        subject_storage_file = "../Data/imu_data_parquet/" + subject_id + "_imu_data.parquet"

        # Check if we've already processed the data for this subject
        if os.path.exists(subject_storage_file):
            print("File %s already exists, skipping subject %s" % (subject_storage_file, subject_id))
            continue

        print("Processing subject %s" % subject_id)
        file_id = 1
        subject_data = []
        for data_file in raw_data_files:
            
            # Which protocol are we processing?
            if subject_id in missing_calibration:
                file_protocol = file_id_mapping_missing[file_id]
            elif subject_id in calibration_last:
                file_protocol = file_id_mapping_last[file_id]
            elif subject_id == "FNP1033":
                file_protocol = file_id_mapping_1033[file_id]
            elif subject_id == "FNP2040":
                file_protocol = file_id_mapping_2040[file_id]
            else:
                file_protocol = file_id_mapping_normal[file_id]
            
            print("Processing %s with file %s" % (file_protocol, os.path.basename(data_file)))

            # Open h5 file for reading
            with h5py.File(data_file, 'r') as f:

                # Get all sensors in the dataset
                sensor_list = f['Sensors']
                
                # Go through each sensor and swap
                for sensor in sensor_list:

                    # Placement is found in label 1
                    label = f['Sensors'][sensor]['Configuration'].attrs['Label 1'].decode('ascii')

                    # Extract the following data and axes from sensor
                    types = ["Accelerometer", "Gyroscope", "Magnetometer"]
                    axes = ["x", "y", "z"]
                    sensor_frame = extract_sensor_into_frame(subject_id, file_protocol, sensor, label, types, axes)
                    subject_data.append(sensor_frame)

                    # Plot acc data
                    plot_data = True
                    if plot_data:
                        acc_data = sensor_frame.iloc[:,0:3].to_numpy()
                        print(acc_data.shape)
                        xaxis = np.arange(0,acc_data.shape[0])
                        plt.plot(xaxis, acc_data[:,0])
                        plt.plot(xaxis, acc_data[:,1])
                        plt.plot(xaxis, acc_data[:,2])
                        plt.show()  
                        
            # Increment file counter
            file_id += 1

        # Concatenate all data for subject and write to parquet format
        subject_frame = pd.concat(subject_data,axis=1)
        print("Storing data in %s" % (subject_storage_file))
        subject_frame.to_parquet(subject_storage_file, engine="pyarrow")

    print("All done")
