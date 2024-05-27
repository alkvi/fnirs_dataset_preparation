import os 
import pandas as pd
import numpy as np
import shutil

if __name__ == "__main__":

    # Where all the data is stored
    snirf_source_folder = "../Data/aurora_fnirs_data"
    subject_folders = [ f.path for f in os.scandir(snirf_source_folder) if "FNP" in f.path]

    # How we will rename sessions/protocols
    session_mapping = {"Setup_1": "protocol1", "Setup_2": "protocol2", "Setup_3": "protocol3"}
    task = "complexwalk"

    # Where we will create the structure for sourcedata2bids
    bids_folder = "../Data/bids_dataset_snirf/sourcedata"

    # Go through each subject folder..
    missing_files = []
    for subject_folder in subject_folders:
        subject_base = os.path.basename(subject_folder)
        subject_bids_name = subject_base.replace("_", "")
        
        # Go through each session
        for session in session_mapping.keys():
            session_folder = f"{subject_folder}/{session}"
            mapped_session = session_mapping[session]

            if not os.path.isdir(session_folder):
                print(f"Session folder {session_folder} not found for subject {subject_base}")
                missing_files.append(session_folder)
                continue

            print(f"Looking in {session_folder}")
            session_snirf_files = [ f.path for f in os.scandir(session_folder) if ".snirf" in f.path]
            if len(session_snirf_files) != 1:
                print(f"ERROR: unexpected amount ({len(session_snirf_files)}) of SNIRF files in folder {session_folder}, please check")
                exit()

            aurora_snirf_file = session_snirf_files[0]
            print(f"Found file {aurora_snirf_file}")
            bids_file = f"sub-{subject_bids_name}_ses-{mapped_session}_task-{task}_run-1_fnirs.snirf"

            # Create subject/session/nirs folder, and place SNIRF in there
            bids_subject_folder = f"{bids_folder}/sub-{subject_bids_name}"
            session_folder = f"{bids_folder}/sub-{subject_bids_name}/ses-{mapped_session}"
            dest_nirs_folder = f"{bids_folder}/sub-{subject_bids_name}/ses-{mapped_session}/nirs"
            print("Creating folder " + dest_nirs_folder)
            if not os.path.isdir(bids_subject_folder):
                os.mkdir(bids_subject_folder)
            os.mkdir(session_folder)
            os.mkdir(dest_nirs_folder)

            # Copy SNIRF
            dest_file = f"{dest_nirs_folder}/{bids_file}"
            print("Copying file to " + dest_file)
            shutil.copyfile(aurora_snirf_file, dest_file)

    print("Done")
    print("Missing files:")
    print(missing_files)