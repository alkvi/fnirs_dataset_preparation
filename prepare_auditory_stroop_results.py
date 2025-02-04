import pandas as pd 
import numpy as np 

# Read auditory stroop correction sheet data
def get_sheet_data(sheet_data, codes, protocol, start_idx):

    # Which ones are single task and which are dual task?
    num_prompts = 7
    if protocol == "protocol_1":
        st_blocks = [1, 3, 6, 8, 9, 11]
        dt_blocks = [2, 4, 5, 7, 10, 12]
        block_types = ["ST", "DT", "All"]
    else:
        st_blocks = []
        dt_blocks = [1, 2, 3, 4, 5, 6]
        block_types = ["DT"]
    
    # Go subject by subject
    subject_data = []
    for idx in range(start_idx,len(sheet_data)):
        answers = sheet_data.iloc[idx].values[1:]
        subject = sheet_data.iloc[idx].values[0]
        answer_codes = pd.DataFrame(data={"codes": codes, "answers": answers})
        answer_codes['block_num'] = np.divide(answer_codes.index.tolist(), num_prompts).astype(int) + 1
        answer_codes['block_type'] = np.where(np.isin(answer_codes['block_num'].values, st_blocks), "ST", "DT")

        # Separate by block type
        for block_type in block_types:
            if sheet_data.iloc[idx].isnull().values.any():
                percent_ic = None
                percent_c = None
                percent_total = None
            else:
                # Get data for only this specific block type
                answer_codes_block = answer_codes[answer_codes['block_type'] == block_type]
                if block_type == "All":
                    answer_codes_block = answer_codes

                # Sum answers
                sum_ic = answer_codes_block[(answer_codes_block['answers'] == 1) & (answer_codes_block['codes'] == "IC")].sum()
                sum_c = answer_codes_block[(answer_codes_block['answers'] == 1) & (answer_codes_block['codes'] == "C")].sum()
                sum_total = answer_codes_block[(answer_codes_block['answers'] == 1)].sum()
                len_ic = len(answer_codes_block[answer_codes_block['codes'] == "IC"])
                len_c = len(answer_codes_block[answer_codes_block['codes'] == "C"])
                len_total = len(answer_codes_block)
                percent_ic = sum_ic['answers'] / len_ic * 100
                percent_c = sum_c['answers'] / len_c * 100
                percent_total = sum_total['answers'] / len_total * 100
                
            # Create dataframe for this subject/protocol/block type
            acc = {"subject":  [subject],
                    "protocol": [protocol],
                    "block_type": [block_type],
                    "accuracy_ic": [percent_ic], 
                    "accuracy_c": [percent_c],
                    "accuracy_total": [percent_total]}
            frame = pd.DataFrame(data=acc)
            subject_data.append(frame)

    return subject_data

# Structure answer times by subject, protocol, block type
def structure_answer_times(answer_times, protocol, codes):

    # Which ones are single task and which are dual task?
    num_prompts = 7
    if protocol == "protocol_1":
        st_blocks = [1, 3, 6, 8, 9, 11]
        dt_blocks = [2, 4, 5, 7, 10, 12]
        block_types = ["ST", "DT", "All"]
    else:
        st_blocks = []
        dt_blocks = [1, 2, 3, 4, 5, 6]
        block_types = ["DT"]

    # Melt into long-form and add info about blocks and congruency
    answer_times['subject'] = "FNP" + answer_times['id'].astype(str)
    answer_times = answer_times.drop(['id', 'protocol'], axis=1)
    answer_times = answer_times.melt(id_vars=['subject'])
    answer_times = answer_times.rename(columns={"variable": "answer", "value": "answer_time"}, errors="raise")
    answer_times['answer_number'] = answer_times['answer'].str.split("_").str[-1].astype(int) - 1
    answer_times['prompt_number'] = answer_times['answer_number'] + 1
    answer_times['congruency'] = codes[answer_times['answer_number']]
    answer_times['block_num'] = np.divide(answer_times['answer_number'].tolist(), num_prompts).astype(int) + 1
    answer_times['block_type'] = np.where(np.isin(answer_times['block_num'].values, st_blocks), "ST", "DT")
    answer_times['protocol'] = protocol

    # Re-order and return
    answer_times = answer_times[['subject', 'protocol', 'block_num', 'block_type', 'prompt_number', 'congruency', 'answer_time']]
    return answer_times

if __name__ == "__main__":

    pd.set_option('display.max_rows', None)

    # Read correction sheet data
    # 0 wrong
    # 1 correct
    # 99 missing answer 
    p1_data_hc = pd.read_excel("temp_data/Accuracy_setup_1_bigstudy.xlsx", sheet_name='SETUP 1_HC')
    p1_data_pd = pd.read_excel("temp_data/Accuracy_setup_1_bigstudy.xlsx", sheet_name='SETUP 1_PD')
    p3_data_hc = pd.read_excel("temp_data/Accuracy_setup_3_bigstudy.xlsx", sheet_name='SETUP 3_HC')
    p3_data_pd = pd.read_excel("temp_data/Accuracy_setup_3_bigstudy.xlsx", sheet_name='SETUP 3_PD')

    # Get congruent/incongruent
    codes_p1 = p1_data_hc.iloc[0].values[1:]
    codes_p3 = p3_data_hc.iloc[0].values[1:]

    # Read answer times
    p1_hc_answer_times = pd.read_csv("temp_data/reaction_times_hc_setup1.csv")
    p3_hc_answer_times = pd.read_csv("temp_data/reaction_times_hc_setup3.csv")
    p1_pd_answer_times = pd.read_csv("temp_data/reaction_times_pd_setup1.csv")
    p3_pd_answer_times = pd.read_csv("temp_data/reaction_times_pd_setup3.csv")

    # Structure answer times
    all_data = []
    all_data.append(structure_answer_times(p1_hc_answer_times, "protocol_1", codes_p1))
    all_data.append(structure_answer_times(p3_hc_answer_times, "protocol_3", codes_p3))
    all_data.append(structure_answer_times(p1_pd_answer_times, "protocol_1", codes_p1))
    all_data.append(structure_answer_times(p3_pd_answer_times, "protocol_3", codes_p3))
    answer_time_frame = pd.concat(all_data)
    answer_time_frame.to_csv("auditory_stroop_answer_time.csv", index=False)
    print(answer_time_frame)

    # Extract relevant data about accuracy
    all_data = []
    all_data.append(get_sheet_data(p1_data_hc, codes_p1, "protocol_1", 1))
    all_data.append(get_sheet_data(p1_data_pd, codes_p1, "protocol_1", 0))
    all_data.append(get_sheet_data(p3_data_hc, codes_p3, "protocol_3", 1))
    all_data.append(get_sheet_data(p3_data_pd, codes_p3, "protocol_3", 0))
    all_data = [item for subject_data in all_data for item in subject_data]
    accuracy_frame = pd.concat(all_data)
    accuracy_frame.to_csv("auditory_stroop_accuracy.csv", index=False)
    print(accuracy_frame)

   


