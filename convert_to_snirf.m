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

%% Rename and insert modified stims into data

% Our design: stimulus length is 20 seconds. Rest is ~15 seconds.
% Protocol 1 has 6*3 conditions, protocol 2&3 have 6*2 conditions.
% The NIRS toolbox expects stimuli to only demarcate start of each
% stimulus. Each stimulus then has a certain length.

% Visualize results before change.
figure(); raw_data(1).draw;
figure(); raw_data(2).draw;
figure(); raw_data(3).draw;

job = nirs.modules.RenameStims;
job.listOfChanges = {
    'stim_channel1', 'Rest';
    'stim_channel2', 'Straight_walking';
    'stim_channel3', 'Stand_still_and_Aud_Stroop';
    'stim_channel4', 'Straight_walking_and_Aud_Stroop';
    'stim_channel5', 'Navigated_walking';
    'stim_channel6', 'Navigation_and_Aud_Stroop'};
raw_data = job.run(raw_data);

% Remove stim 1 (rest).
j=nirs.modules.DiscardStims;
j.listOfStims={'Rest'};
raw_data = j.run(raw_data);

% Set each duration to 20 seconds.
raw_data = nirs.design.change_stimulus_duration(raw_data,[],20);

% Visualize results after change.
figure(); raw_data(1).draw;
figure(); raw_data(2).draw;
figure(); raw_data(3).draw;

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
