% Original version by lucian bezuidenhout
% Modified by alkvis, 2023-10-27
% This code is designed to calculate answer times in auditory stroop data.

tic
clear all
clc
close all
workspace;  % Make sure the workspace panel is showing.
format long g;  
format compact;
fontSize = 20;

%% Settings

file_dir = "C:\fnirs_audio\WAV_filer\PD\Setup3";
files = dir(file_dir + "\*.wav");
protocol = "setup3";
plot_figure = false;
response_threshold_multiplier = 8.0;

% Participants to skip (e.g. missing protocol)
%participant_exclude = [1044];
%participant_exclude = [];
participant_exclude = [2010];

% These are the start times of stimuli in each protocol (in seconds and sample).
fs = 44100;
if strcmp(protocol, "setup1")
    stim_start_s = [55, 94, 132, 208, 284, 322, 361, 399, 512, 551, 588, 664];
    stim_start_idx = stim_start_s * fs;
    expected_stims = 84;
else
    stim_start_s = [18, 94, 208, 284, 322, 399];
    stim_start_idx = stim_start_s * fs;
    expected_stims = 42;
end

%% Prepare a table

result_file_name = "reaction_times_pd_setup3.csv";
if isfile(result_file_name)
    full_participant_table = readtable(result_file_name);
    full_participant_table = mergevars(full_participant_table, [3:expected_stims+2], "NewVariableName","answer");
else
    full_participant_table = table();
    full_participant_table.id = zeros(0);
    full_participant_table.protocol = zeros(0);
    full_participant_table.answer = zeros(0);  
end

%% Go through each file and get file names

file_names = [];
for idx=1:length(files)
    file_names = [file_names; convertCharsToStrings(files(idx).name)];
end

%% Load data

for idx=1:length(file_names)

    filename = file_names(idx);
    filepath = fullfile(file_dir, filename);

    fprintf("Processing %s\n", filename);

    subject_id = split(filename,"_");
    subject_id = erase(subject_id(2),"fnp");
    subject_id = erase(subject_id,".wav");
    
    % Have we processed this file?
    if sum(full_participant_table.id == str2num(subject_id))
        fprintf("Already processed %s, skipping\n", subject_id);
        continue
    end

    % Have we specified to skip this participant?
    if sum(participant_exclude == str2num(subject_id))
        fprintf("Specified to skip %s, skipping\n", subject_id);
        continue
    end
    
    % With SyncBox, stim is a bit longer in
    if str2num(subject_id) >= 1029
       stim_start_s = stim_start_s + 5;
       stim_start_idx = stim_start_s * fs; 
    end

    %Import the audiofile
    [raw_audio, fs] = audioread(filepath);  

    % We can get the end points by taking the starts and adding 20s.
    stim_end_idx = stim_start_idx + (20 * fs);
    cut_end_idx = stim_end_idx(end) + (2 * fs);

    % Zero any signal after end of protocol
    raw_audio(cut_end_idx:end,:) = 0;

    %combine the audio and response signal
    audio_signal = raw_audio(:,1);
    response_signal = raw_audio(:,2);

    % Get time vectors in seconds
    aud_time = (1:length(audio_signal))./fs;
    res_time = (1:length(response_signal))./fs;

    %% Envelope stimuli

    %Find the envelope by taking a moving max operation, imdilate.
    envelope_audio_signal = imdilate(abs(audio_signal), true(10000, 1));
    audio_thres = (max(audio_signal))/3;

    %seperate the signal into individual responses
    quiet_parts_audio_signal = envelope_audio_signal > audio_thres;
    idx_prompt_start = strfind(quiet_parts_audio_signal',[0 1]);
    idx_prompt_end = strfind(quiet_parts_audio_signal', [1 0]);

    % Verify we have the correct number of prompts
    if (length(idx_prompt_start) ~= expected_stims) || (length(idx_prompt_end) ~= expected_stims)
        disp("WARN: unexpected number of stimuli");
        pause;
    end

    %% Envelope response 

    %filter the response signal 
    % Now smooth with a Savitzky-Golay sliding polynomial filter
    windowWidth = 401; % window being filtered
    polynomialOrder = 20; %polynomial order
    filter_response = sgolayfilt(response_signal, polynomialOrder, windowWidth);

    %Find the envelope by taking a moving max operation, imdilate.
    envelope_response_signal = imdilate(abs(filter_response), true(15000, 1));

    %Determine threshold
    AR = envelope_response_signal(envelope_response_signal>=0);
    response_thres = mean(AR)*response_threshold_multiplier;

    %seperate the signal into individual responses
    quietParts_response_signal = envelope_response_signal > response_thres;
    idx_response_start = strfind(quietParts_response_signal',[0 1]);
    idx_response_end = strfind(quietParts_response_signal', [1 0]);

    %% Plot track with beginning and ends

    if plot_figure
        figure;
        subplot(2,1,1);
        plot(aud_time, audio_signal');
        hold on;
        xline(aud_time(idx_prompt_start), 'g');
        xline(aud_time(idx_prompt_end), 'r');
        yline(audio_thres, '--k');
        yline(-audio_thres, '--k');
        xline(aud_time(stim_start_idx(1)), 'm', 'LineWidth',5);
        xline(aud_time(stim_end_idx(end)), 'm', 'LineWidth',5);
        hold off;

        subplot(2,1,2);
        plot(res_time, filter_response');
        hold on;
        xline(res_time(idx_response_start), 'g');
        xline(res_time(idx_response_end), 'r');
        yline(response_thres, '--k');
        yline(-response_thres, '--k');
        xline(res_time(stim_start_idx(1)), 'm', 'LineWidth',5);
        xline(res_time(stim_end_idx(end)), 'm', 'LineWidth',5);
        hold off;
    end

    %% Identify missing responses, or out of order responses

    if (length(idx_response_start) ~= expected_stims) || (length(idx_response_end) ~= expected_stims)
        disp("INFO: there are missing responses");
    end

    % Detect missing responses in a search interval.
    % Consider stimuli/prompt to response, with appropriate 
    % buffer. If nothing inbetween prompt - prompt, then missing answer.
    missing_response_idx = [];

    for prompt_idx = 1:length(idx_prompt_start)-1
        search_interval_start = idx_prompt_start(prompt_idx);
        stim_length = idx_prompt_end(prompt_idx) - idx_prompt_start(prompt_idx);

        % Search until next prompt, with some buffer
        search_interval_end = idx_prompt_start(prompt_idx+1) + 0.5*fs;

        % see if a response exists in the interval
        response_exists = (idx_response_start > search_interval_start) & (idx_response_start < search_interval_end);
        if sum(response_exists) < 1
            missing_response_idx = [missing_response_idx, prompt_idx];
        end
    end

    % Did we catch all the missing responses?
    if length(idx_response_start) + length(missing_response_idx) ~= expected_stims
        disp("INFO: there still are missing responses");
        plot_figure = 1;
        pause
    end

    %% Finally get delay

    % Convert from index to actual times in seconds
    time_prompt_start = aud_time(idx_prompt_start);
    time_prompt_end = aud_time(idx_prompt_end);
    time_response_start = res_time(idx_response_start);
    time_response_end = res_time(idx_response_end);

    % Add NaN at missing response indices
    for m_idx=1:length(missing_response_idx)
        missing_idx = missing_response_idx(m_idx);
        if missing_idx == 1
            time_response_start = [NaN time_response_start];
        elseif missing_idx == expected_stims
            time_response_start = [time_response_start NaN];
        else
            time_response_start = [time_response_start(1:missing_idx-1), NaN, time_response_start(missing_idx:end)];
        end
    end

    % Get the time from start of prompt to start of answer
    answer_times = time_response_start - time_prompt_start;

    %% Add to table

    result_table = table();
    result_table.id = str2num(subject_id);
    result_table.protocol = protocol;
    result_table.answer = answer_times;

    full_participant_table = [full_participant_table; result_table];
    writetable(full_participant_table,result_file_name)

end

disp("All done");
