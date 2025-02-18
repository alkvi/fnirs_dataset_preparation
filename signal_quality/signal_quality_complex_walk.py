import os
import mne
import pickle
import json
import numpy as np
import pandas as pd
from itertools import compress
from mne.preprocessing.nirs import optical_density
from mne_bids import BIDSPath, read_raw_bids

if __name__ == "__main__":

    # Specify BIDS root folder
    bids_root = "../bids_dataset_snirf"

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
    suffix = 'nirs'
    sessions = ['protocol1', 'protocol2', 'protocol3']

    # Save data in dicts for each protocol/condition
    all_sci_per_condition = {}
    all_bad_channels_per_participant = {}

    # Go through data for each subject
    for subject in all_subjects:
        for session in sessions:

            # Load
            bids_path = BIDSPath(subject=subject, task=task, session=session,
                                suffix=suffix, datatype=datatype, root=bids_root)
            print("Using BIDS file path..")
            print(bids_path)
            if not os.path.exists(bids_path):
                print("No SNIRF file, skip: " + str(bids_path))
                continue
            raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)

            # Get stims
            annotations = raw_intensity.annotations
            annotations.set_durations(18)
            raw_intensity.set_annotations(annotations)

            # Which channels are short channels? Have 16 (8 for 850, 8 for 760)
            short_idx = mne.preprocessing.nirs.short_channels(raw_intensity.info)
            short_idx = np.where(short_idx)[0]
            long_ch_names = np.array(raw_intensity.ch_names)
            np.delete(long_ch_names, short_idx)

            # Then get the indices of annotations with certain descriptions, then get chunks and calculate SCI values per condition
            try:
                condition_data = raw_intensity.crop_by_annotations()
                for chunk in condition_data:
                    chunk.load_data()
                    chunk_condition = chunk.annotations.description
                    print(chunk_condition)
                    if len(chunk_condition) > 1:
                        print("ERROR: expected one condition per chunk")
                        #exit()
                    chunk_condition = chunk_condition[0]
                    chunk_od = optical_density(chunk)
                    sci = mne.preprocessing.nirs.scalp_coupling_index(chunk_od)
                    bad_channels = list(compress(chunk_od.ch_names, sci < 0.7))

                    # Create key to store SCI for protocol / condition, initialize if empty for this condition
                    sci_key = f"{session}-{chunk_condition}"
                    if sci_key not in all_sci_per_condition:
                        all_sci_per_condition[sci_key] = []
                    
                    # Store SCI
                    sci = np.array(sci)
                    sci = np.delete(sci, short_idx, axis=0)
                    all_sci_per_condition[sci_key].append(sci)
            except:
                print("Could not get condition data, continuing")

            # Calculate overall SCI over entire length of signal
            od = optical_density(raw_intensity)
            overall_sci = mne.preprocessing.nirs.scalp_coupling_index(od)
            overall_sci = np.array(overall_sci)
            overall_sci_long = np.delete(overall_sci, short_idx, axis=0)

            # Get bad channels for subject/session
            bad_channels = list(compress(raw_intensity.ch_names, overall_sci < 0.7))
            bad_channels = np.unique([channel.replace(" 760", "").replace(" 850", "") for channel in np.unique(bad_channels)]).tolist()

            bad_channel_key = f"{subject}-{session}"
            if bad_channel_key not in all_bad_channels_per_participant:
                all_bad_channels_per_participant[bad_channel_key] = []
            if len(bad_channels) > 0:
                all_bad_channels_per_participant[bad_channel_key] = bad_channels
            
# Save pickles     
with open('all_sci_per_condition.pickle', 'wb') as handle:
    pickle.dump(all_sci_per_condition, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('all_bad_channels_per_participant.pickle', 'wb') as handle:
    pickle.dump(all_bad_channels_per_participant, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("SCI per protocol / condition")
print(all_sci_per_condition)

all_sci_df = []
for key in all_sci_per_condition:
    print(f"\nKey: {key}")
    sci_for_condition = np.concatenate(all_sci_per_condition[key])

    print("\nMean SCI value")
    print(np.mean(sci_for_condition))

    print("\nMean SCI values, SD")
    print(np.std(sci_for_condition))

    sci_vals_bad = sci_for_condition[sci_for_condition<=0.7]
    print("\nCount of bad SCI values (<0.7)")
    print(len(sci_vals_bad))

    sci_df = pd.DataFrame({
    'protocol': [key.split("-")[0]],
    'condition': [key.split("-")[1]],
    'sci': [np.mean(sci_for_condition)],
    'sci_std': [np.std(sci_for_condition)],
    'bad_channels': [len(sci_vals_bad)]
    })
    all_sci_df.append(sci_df)

all_sci_df = pd.concat(all_sci_df)
all_sci_df.to_csv("signal_quality_sci.csv", index=False)

# Prepare a JSON with bad channels
with open('bad_channels.json', 'w') as handle:
    json.dump(all_bad_channels_per_participant, handle)