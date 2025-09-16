% Data preparation
% Ran on R2024a, forked version of NIRS toolbox
% https://github.com/alkvi/nirs-toolbox-fork/tree/phd_study_3
%% Load data

% Load all data. Directory is structured as group/subject/session.
my_data_dir = 'original_format_nirx_data';
raw_data = nirs.io.loadDirectory(my_data_dir,{'group','subject', 'session'}); 

% Check loaded demographics.
demographics = nirs.createDemographicsTable(raw_data);
disp(demographics);

% Save data as intermediary step.
save('raw_data.mat','raw_data');

%% Modify demographics to BIDS compliant standard

old_names = ["Setup_1" "Setup_2" "Setup_3"];
new_names = ["protocol1" "protocol2" "protocol3"];
map = containers.Map(old_names,new_names);

for i=1:size(raw_data,1)
    old_subject = raw_data(i).demographics('subject');
    new_subject = strrep(old_subject, "_", "");
    raw_data(i).demographics('subject') = convertStringsToChars(new_subject);
    
    old_session = raw_data(i).demographics('session');
    new_session = map(old_session);
    raw_data(i).demographics('session') = new_session;
end

demographics = nirs.createDemographicsTable(raw_data);
disp(demographics);

%% Fix stimuli

% Our design: stimulus length is 20 seconds. Rest is ~15 seconds.
% Protocol 1 has 6*3 conditions, protocol 2&3 have 6*2 conditions.
% Triggers arrive as: (1) start rest period (2) start block period.
% The NIRS toolbox expects stimuli to only demarcate start of each
% stimulus. Each stimulus then has a certain length.
% We therefore need to remove the trigger for the start of the rest period
% from our data.

stim_table = nirs.createStimulusTable(raw_data);

%%  Manual trigger fixes

% Index 237 (a protocol 2 session) has 24 manual triggers (all in channel 1).
% Rest, block, repeat 12 times. Copy to each other channel.
% Then keep only corresponding ones. One trigger was 8 seconds late.
row_index = 237;
stim_table.stim_channel1{row_index,1}.onset(7) = stim_table.stim_channel1{row_index,1}.onset(7) - 8;
stim_table(row_index,3:13) = stim_table(row_index,2);
stim_table = populate_manual_stims(row_index, stim_table, 13);

% Index 267 (p1), 268 (p2), 269 (p3) were also manually marked. 
% These are in channel 5.
row_index = 267;
stim_table(row_index,2:19) = stim_table(row_index,6);
stim_table = populate_manual_stims(row_index, stim_table, 19);

row_index = 268;
stim_table(row_index,2:13) = stim_table(row_index,6);
stim_table = populate_manual_stims(row_index, stim_table, 13);

row_index = 269;
stim_table(row_index,2:13) = stim_table(row_index,6);
stim_table = populate_manual_stims(row_index, stim_table, 13);

%% Fixes to all stimuli

for row_index = 1:size(stim_table,1)
    
    % Protocol 1 contains 18 stimuli (each condition 6 times).
    % The stim_table contains 19 stimuli; the last one should be removed (we
    % have an extra block in PsychoPy at the end after experiment is over). The
    % last one represents start of final rest+start of extra block.
    % Same for protocol 2 and 3, but these have 12 stimuli, so remove stim 13.
    if strcmp(demographics.session{row_index}, "protocol1")
        stim_table(row_index,20) = {''};
        max_col = 19;
    else
        stim_table(row_index,14) = {''};
        max_col = 13;
    end
    
    % Index 64 lacks one trigger.
    if row_index == 64
        max_col = 12;
    end
    
    % Index 227 only has a block of recorded data.
    if row_index == 227
        max_col = 3;
    end
    
    % Remove the first trigger marking the start of rest period.
    for column_index = 2:max_col
        % Becomes cell array after column 14
        if iscell(stim_table{row_index,column_index})
            stim_table{row_index,column_index}{1}.onset(1) = [];
            stim_table{row_index,column_index}{1}.dur(1) = []; 
        else
            stim_table{row_index,column_index}.onset(1) = [];
            stim_table{row_index,column_index}.dur(1) = []; 
        end
    end
    
end

%% Rename and insert modified stims into data

% Visualize results before change.
figure(); raw_data(1).draw;
figure(); raw_data(2).draw;
figure(); raw_data(3).draw;

for row_index=1:size(raw_data,1)
    
    if strcmp(demographics.session{row_index}, "protocol1")
        stim_names = ["stim_channel1", "stim_channel2", "stim_channel3", ... 
                        "stim_channel4", "stim_channel5" "stim_channel6" ...
                        "stim_channel7", "stim_channel8" "stim_channel9" ...
                        "stim_channel10", "stim_channel11" "stim_channel12" ...
                        "stim_channel13", "stim_channel14" "stim_channel15" ...
                        "stim_channel16", "stim_channel17" "stim_channel18"];
        stim_rename_list = {'stim_channel1', 'Straight_walking';
                         'stim_channel2' ,'Stand_still_and_Aud_Stroop';
                         'stim_channel3' ,'Straight_walking_and_Aud_Stroop';
                         'stim_channel4' ,'Stand_still_and_Aud_Stroop';
                         'stim_channel5' ,'Straight_walking';
                         'stim_channel6' ,'Straight_walking_and_Aud_Stroop';
                         'stim_channel7' ,'Straight_walking';
                         'stim_channel8' ,'Straight_walking_and_Aud_Stroop';
                         'stim_channel9' ,'Stand_still_and_Aud_Stroop';
                         'stim_channel10' ,'Straight_walking_and_Aud_Stroop';
                         'stim_channel11' ,'Stand_still_and_Aud_Stroop';
                         'stim_channel12' ,'Straight_walking'
                         'stim_channel13' ,'Straight_walking'
                         'stim_channel14' ,'Stand_still_and_Aud_Stroop'
                         'stim_channel15' ,'Straight_walking_and_Aud_Stroop'
                         'stim_channel16' ,'Stand_still_and_Aud_Stroop'
                         'stim_channel17' ,'Straight_walking'
                         'stim_channel18' ,'Straight_walking_and_Aud_Stroop'};
        stim_discard = {'stim_channel19'};
        stim_table_discard_idx = 20;
    elseif strcmp(demographics.session{row_index}, "protocol2")
        stim_rename_list = {'stim_channel1', 'Straight_walking';
                         'stim_channel2' ,'Navigated_walking';
                         'stim_channel3' ,'Straight_walking';
                         'stim_channel4' ,'Navigated_walking';
                         'stim_channel5' ,'Navigated_walking';
                         'stim_channel6' ,'Straight_walking';
                         'stim_channel7' ,'Navigated_walking';
                         'stim_channel8' ,'Straight_walking';
                         'stim_channel9' ,'Straight_walking';
                         'stim_channel10' ,'Navigated_walking';
                         'stim_channel11' ,'Straight_walking';
                         'stim_channel12' ,'Navigated_walking'};
        stim_names = ["stim_channel1", "stim_channel2", "stim_channel3", ... 
                        "stim_channel4", "stim_channel5" "stim_channel6" ...
                        "stim_channel7", "stim_channel8" "stim_channel9" ...
                        "stim_channel10", "stim_channel11" "stim_channel12"];
        stim_discard = {'stim_channel13'};
        stim_table_discard_idx = 14;
    else
        stim_names = ["stim_channel1", "stim_channel2", "stim_channel3", ... 
                        "stim_channel4", "stim_channel5" "stim_channel6" ...
                        "stim_channel7", "stim_channel8" "stim_channel9" ...
                        "stim_channel10", "stim_channel11" "stim_channel12"];
        stim_rename_list = {'stim_channel1', 'Navigation_and_Aud_Stroop';
                         'stim_channel2' ,'Navigation';
                         'stim_channel3' ,'Navigation_and_Aud_Stroop';
                         'stim_channel4' ,'Navigation';
                         'stim_channel5' ,'Navigation';
                         'stim_channel6' ,'Navigation_and_Aud_Stroop';
                         'stim_channel7' ,'Navigation';
                         'stim_channel8' ,'Navigation_and_Aud_Stroop';
                         'stim_channel9' ,'Navigation_and_Aud_Stroop';
                         'stim_channel10' ,'Navigation';
                         'stim_channel11' ,'Navigation_and_Aud_Stroop';
                         'stim_channel12' ,'Navigation'};
        stim_discard = {'stim_channel13'};
        stim_table_discard_idx = 14;
    end
    
    % Index 64 lacks one trigger.
    if row_index == 64
        stim_names = stim_names(1:end-1);
        stim_table_discard_idx = 13;
    end
    
    % Index 227 only has a block of recorded data.
    if row_index == 227
        stim_names = stim_names(1:2);
        stim_table_discard_idx = 3;
    end
    
    % Remove the extra trigger at the end.
    stim_row = stim_table(row_index,:);
    if size(stim_row,2) >= stim_table_discard_idx
        stim_row(:,stim_table_discard_idx:end) = [];
    end
    
    job = nirs.modules.DiscardStims;
    job.listOfStims = stim_discard;
    raw_data(row_index) = job.run(raw_data(row_index));
    
    % Run the toolbox job to create the stimulus data.
    % fileIdx=newstiminfo.FileIdx(idx);
    job = nirs.modules.ChangeStimulusInfo();
    job.ChangeTable = stim_row;
    raw_data(row_index) = job.run(raw_data(row_index));
    
    % Set each duration to 20 seconds.
    raw_data(row_index) = nirs.design.change_stimulus_duration(raw_data(row_index),stim_names,20);
    
    % Rename the stimuli
    job = nirs.modules.RenameStims;
    job.listOfChanges = stim_rename_list;
    raw_data(row_index) = job.run(raw_data(row_index));
end

% Visualize results after change.
figure(); raw_data(1).draw;
figure(); raw_data(2).draw;
figure(); raw_data(3).draw;

% figure(); raw_data(64).draw;
% figure(); raw_data(227).draw;
% figure(); raw_data(237).draw;
% 
% figure(); raw_data(267).draw;
% figure(); raw_data(268).draw;
% figure(); raw_data(269).draw;

%% Convert to BIDS & SNIRF

output_folder_single = "nirs_toolbox_snirf_output";

for row_index=1:size(raw_data,1)

    subj_id = strcat('sub-', demographics.subject{row_index});
    sess = demographics.session{row_index};
    task = 'complexwalk';
    nirs.bids.Data2BIDS(raw_data(row_index), output_folder_single, subj_id, sess, task);

end

% The DataSet2BIDS function isn't quite ready but could be handy 
% in the future
%output_folder = "test_output";
%nirs.bids.DataSet2BIDS(raw_data, output_folder);

%% Helper functions

function stim_table = populate_manual_stims(row_index, stim_table, max_col_idx)

stim_start_idx = 1;
for column_index=2:max_col_idx
    orig_onset = stim_table{row_index,column_index}{1}.onset;
    orig_dur = stim_table{row_index,column_index}{1}.dur;
    orig_amp = stim_table{row_index,column_index}{1}.amp;
    new_onset = orig_onset(stim_start_idx:stim_start_idx+1);
    new_dur = orig_dur(stim_start_idx:stim_start_idx+1);
    new_amp = orig_amp(stim_start_idx:stim_start_idx+1);
    stim_table{row_index,column_index}{1}.onset = new_onset;
    stim_table{row_index,column_index}{1}.dur = new_dur;
    stim_table{row_index,column_index}{1}.amp = new_amp;
    stim_start_idx = stim_start_idx + 2;
end

end
