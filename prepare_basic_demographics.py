import os 
import pandas as pd 
import numpy as np 

if __name__ == "__main__":

    pd.set_option('display.max_rows', None)

    crf_file = "temp_data/CRF_data.csv"
    crf_data = pd.read_csv(crf_file)

    # Prepare identifier lists
    data_ms  = crf_data[crf_data['participant_id'].str.contains('FNP3')]
    id_list = data_ms['participant_id'].str.extract(r'(FNP.{4})')
    data_ms['participant_id'] = id_list
    data_ms['participant_id'].to_csv("temp_data/identifiers_MS.csv", index=False)

    data_hc  = crf_data[crf_data['participant_id'].str.contains('FNP1')]
    id_list = data_hc['participant_id'].str.extract(r'(FNP.{4})')
    data_hc['participant_id'] = id_list
    data_hc['participant_id'].to_csv("temp_data/identifiers_HC.csv", index=False)

    # Get the identifiers
    hc_ids = pd.read_csv("temp_data/identifiers_HC.csv")['participant_id'].to_list()
    ms_ids = pd.read_csv("temp_data/identifiers_MS.csv")['participant_id'].to_list()

    # Construct df from filtered
    crf_data = pd.concat([data_hc, data_ms], axis=0, ignore_index=True)

    # Assign groups
    ids = sorted(list(set(crf_data['participant_id'].to_list())))
    for id in hc_ids:
        crf_data.loc[crf_data.participant_id == id, 'group'] = "HC"
    for id in ms_ids:
        crf_data.loc[crf_data.participant_id == id, 'group'] = "MS"

    # Keep some basic demographic data for demographics file
    crf_data = crf_data[['participant_id', 'group', 'crf_phone_gender', 'crf_phone_age', 'crf_length', 'crf_weight', 'crf_frandin', 
                         'crf_ms_diagnosis', 'crf_year_ms_diagnosis', 'crf_ms_onset']]
    
    crf_data = crf_data.rename(columns={"participant_id": "subject", 
                                        "crf_phone_gender": "sex", 
                                        "crf_phone_age": "age", 
                                        "crf_length": "height",
                                        "crf_weight": "weight", 
                                        "crf_frandin": "frandin_grimby", 
                                        "crf_ms_diagnosis": "ms_diagnosis", 
                                        "crf_year_ms_diagnosis": "ms_diagnosis_year",
                                        "crf_ms_onset": "ms_onset_year"})
    
    print(crf_data)
    crf_data.to_csv("temp_data/basic_demographics.csv", index=False)