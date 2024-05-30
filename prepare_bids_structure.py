import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time 
import shutil

if __name__ == "__main__":

    # Where all the data is stored
    snirf_source_folder = "../Data/nirs_toolbox_snirf_output"
    files = [ f.path for f in os.scandir(snirf_source_folder) if ".snirf" in f.path]

    # Where we will create the structure for sourcedata2bids
    bids_folder = "../Data/bids_dataset_snirf/sourcedata"

    for file in files:
        print("Processing file " + file)
        file_base = os.path.basename(file)
        bids_parts = file_base.split('_')
        subject = bids_parts[0]
        session = bids_parts[1]

        # Create subject/session/nirs folder, and place SNIRF in there
        subject_folder = bids_folder + "/" + subject
        session_folder = bids_folder + "/" + subject + "/" + session
        dest_nirs_folder = bids_folder + "/" + subject + "/" + session + "/nirs"
        print("Creating folder " + dest_nirs_folder)
        if not os.path.isdir(subject_folder):
            os.mkdir(subject_folder)
        os.mkdir(session_folder)
        os.mkdir(dest_nirs_folder)

        # Copy SNIRF
        dest_file = dest_nirs_folder + "/" + file_base
        print("Copying file to " + dest_file)
        shutil.copyfile(file, dest_file)
        
    print("Done")