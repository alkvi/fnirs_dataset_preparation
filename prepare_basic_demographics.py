import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

if __name__ == "__main__":

    pd.set_option('display.max_rows', None)

    crf_file = "../Data/REDcap_data/CRF_data.csv"
    crf_data = pd.read_csv(crf_file)

    # Prepare identifier lists
    data_pd  = crf_data[crf_data['id_nummer'].str.contains('FNP2')]
    data_pd['id_nummer'].to_csv("../Data/identifiers_PD.csv", index=False)

    data_hc  = crf_data[crf_data['id_nummer'].str.contains('FNP1')]
    data_old = data_hc[data_hc['lder_phone'] >= 60]
    data_old['id_nummer'].to_csv("../Data/identifiers_OA.csv", index=False)

    data_hc  = crf_data[crf_data['id_nummer'].str.contains('FNP1')]
    data_young = data_hc[data_hc['lder_phone'] <= 50]
    data_young['id_nummer'].to_csv("../Data/identifiers_YA.csv", index=False)

    # Get the identifiers
    ya_ids = pd.read_csv("../Data/identifiers_YA.csv")['id_nummer'].to_list()
    oa_ids = pd.read_csv("../Data/identifiers_OA.csv")['id_nummer'].to_list()
    pd_ids = pd.read_csv("../Data/identifiers_PD.csv")['id_nummer'].to_list()

    # Assign groups
    ids = sorted(list(set(crf_data['id_nummer'].to_list())))
    for id in ya_ids:
        crf_data.loc[crf_data.id_nummer == id, 'group'] = "YA"
    for id in oa_ids:
        crf_data.loc[crf_data.id_nummer == id, 'group'] = "OA"
    for id in pd_ids:
        crf_data.loc[crf_data.id_nummer == id, 'group'] = "PD"

    # Keep some basic demographic data for demographics file
    crf_data = crf_data[['id_nummer', 'group', 'k_n_phone', 'lder_phone', 'vad_r_din_l_ngd_i_cm', 'vad_r_din_vikt_i_kg', 'frandin_grimby', 
                         'crf_pd_phone', 'crf_pd_year_phone', 'crf_pd_debut_phone']]
    
    crf_data = crf_data.rename(columns={"id_nummer": "subject", 
                                        "k_n_phone": "sex", 
                                        "lder_phone": "age", 
                                        "vad_r_din_l_ngd_i_cm": "height",
                                        "vad_r_din_vikt_i_kg": "weight", 
                                        "frandin_grimby": "frandin_grimby", 
                                        "crf_pd_phone": "parkinson_diagnosis", 
                                        "crf_pd_year_phone": "parkinson_debut_year",
                                        "crf_pd_debut_phone": "parkinson_start_symptom_side"})
    
    print(crf_data)
    crf_data.to_csv("../Data/basic_demographics.csv", index=False)