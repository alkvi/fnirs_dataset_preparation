import os
import pandas as pd 
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report

if __name__ == "__main__":

    # Specify BIDS root folder
    bids_root = "../Data/fNIRS_data/bids_dataset_snirf"
    #print(make_report(bids_root))

    # We have 4 file types: events, channels, optodes, and nirs.
    datatype = 'nirs'
    bids_path = BIDSPath(root=bids_root, datatype=datatype)
    nirs_files = bids_path.match()

    # Get all subjects in a sorted list
    all_subjects = [file.subject for file in nirs_files]
    all_subjects = sorted(list(set(all_subjects)))
    print("Subjects")
    print(all_subjects)

    # Prepare params
    task = 'complexwalk'
    suffix = 'events'
    sessions = ['protocol1', 'protocol2', 'protocol3']

    # Go through data for each subject
    event_frames = []
    for subject in all_subjects:
        for session in sessions:
        
            # Get the file for this subject/session
            bids_path = BIDSPath(subject=subject, task=task, session=session,
                                suffix=suffix, datatype=datatype, root=bids_root)
            print("Using BIDS file path..")
            print(bids_path)

            if not os.path.exists(bids_path):
                print("No file, skip: " + str(bids_path))
                continue
            
            events = pd.read_csv(str(bids_path), sep='\t')
            events["subject"] = subject
            events["session"] = session
            events["duration"] = 20
            event_frames.append(events)

    full_frame = pd.concat(event_frames)
    print(full_frame)
    full_frame.to_csv("../Data/temp_data/all_events_nirs.csv", index=False)
