import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

if __name__ == "__main__":

    bids_root = "../bids_dataset_snirf"
    datatype = 'nirs'
    bids_path = BIDSPath(root=bids_root, datatype=datatype)

    # Read an example subject and get a list of optodes
    bids_path = BIDSPath(subject="FNP1001", task="complexwalk", session="protocol1",
                        suffix="nirs", datatype=datatype, root=bids_root)
    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)

    ch_names = np.array(raw_intensity.ch_names)
    ch_names = np.unique([channel.replace(" 760", "").replace(" 850", "") for channel in np.unique(ch_names)]).tolist()
    print(ch_names)

    # Load the bad channels
    bad_channels = pd.read_pickle(r'all_bad_channels_per_participant.pickle')

    all_rows = []
    for ch_name in ch_names:

        print(f"Looking for {ch_name}")
        source = ch_name.split('_')[0].replace("S", "")
        detector = ch_name.split('_')[1].replace("D", "")

        bad_subjects = []
        for key in bad_channels:
            if ch_name in bad_channels[key]:
                bad_subjects.append(key)

        if len(bad_subjects) > 0:
            subj_string = ",".join(bad_subjects)
        else:
            subj_string = "None"
        print(subj_string)

        row_data = {'src': [source], "det": [detector], 'subject': [subj_string]}
        row = pd.DataFrame(row_data)
        all_rows.append(row)

    src_det_table = pd.concat(all_rows)
    print(src_det_table)

# Export.
src_det_table.to_csv('sci_bad_channels.csv', index=False)  

print("Done")
