% This code is taken from QT-NIRS (with some fixed params)
% https://github.com/lpollonini/qt-nirs/blob/71a1da73f8903b56dd0a9829b5498d28f3747022/qtnirs.m
% Hernandez, S. M., & Pollonini, L. (2020, April). NIRSplot: a tool for quality assessment of fNIRS scans. In Optics and the Brain (pp. BM2C-5). Optica Publishing Group.

function power_array = get_power_qt_nirs(data_in, data_types, fcut, window, n_channels, fs)

%raw = getappdata(main_fig,'rawNirs');
%nirsplot_param = getappdata(main_fig,'nirsplot_parameters');
%fcut = nirsplot_param.fcut;
%window = nirsplot_param.window;
%lambda_mask = nirsplot_param.lambda_mask;
%lambdas = nirsplot_param.lambdas;
lambdas = [760 850];
overlap = 0;

% Set the bandpass filter parameters
%fs = 1/mean(diff(raw.t));
%fs = nirsplot_param.fs;
%n_samples = size(raw.d,1);
n_samples = size(data_in,1);
fcut_min = fcut(1);
fcut_max = fcut(2);
if fcut_max >= (fs)/2
    fcut_max = (fs)/2 - eps;
    warning(['The highpass cutoff has been reduced from ',...
        num2str(fcut(2)), ' Hz to ', num2str(fcut_max),...
        ' Hz to satisfy the Nyquist sampling criterion']);
end
[B1,A1]=butter(1,[fcut_min*(2/fs) fcut_max*(2/fs)]);

%nirs_data = zeros(length(lambdas),n_samples,n_channels);
%cardiac_data = zeros(length(raw.SD.Lambda),n_samples,n_channels); % Lambdas x time x channels

nirs_data = zeros(2,n_samples,n_channels);
cardiac_data = zeros(2,n_samples,n_channels); % Lambdas x time x channels

%for j = 1:length(raw.SD.Lambda)
for j = 1:2
    % Filter everything but the cardiac component
    %idx = find(raw.SD.MeasList(:,4) == j);
    if j == 1
        idx = data_types == 760;
    else
        idx = data_types == 850;
    end
    nirs_data(j,:,:) = data_in(:,idx);
    filtered_nirs_data=filtfilt(B1,A1,squeeze(nirs_data(j,:,:)));
    cardiac_data(j,:,:)=filtered_nirs_data./repmat(std(filtered_nirs_data,0,1),size(filtered_nirs_data,1),1); % Normalized heartbeat
end
overlap_samples = floor(window*fs*overlap);
window_samples = floor(window*fs);
if overlap ==0
    n_windows = floor((n_samples)/(window_samples));
else % Valid for overlap=50%
    n_windows = 2*floor((n_samples)/(window_samples))-1;
end
%cardiac_data = cardiac_data(find(lambda_mask),:,:);
sci_array = zeros(size(cardiac_data,3),n_windows);    % Number of optode is from the user's layout, not the machine
power_array = zeros(size(cardiac_data,3),n_windows);
fpower_array = zeros(size(cardiac_data,3),n_windows);
cardiac_windows = zeros(length(lambdas),window_samples,n_channels,n_windows);
for j = 1:n_windows
    if overlap~=0
        if mod(j,2)==0
            jj = floor((j-1)/2)+1;
            interval = (jj-1)*window_samples+1 : jj*window_samples;
            interval = interval + overlap_samples ;
        else
            jj = floor(j/2)+1;
            interval = (jj-1)*window_samples+1 : jj*window_samples;
        end
    else 
       interval = (j-1)*window_samples+1 : j*window_samples;
    end
    cardiac_windows(:,:,:,j) = cardiac_data(:,interval,:);
%             if j<5 || j>(n_windows-4)
%                 disp(['interval(',num2str(j),'):',num2str(interval(1)),'-',num2str(interval(end))]);
%             end
end

for j = 1:n_windows
    cardiac_window = cardiac_windows(:,:,:,j);
    sci_array_channels = zeros(1,size(cardiac_window,3));
    power_array_channels = zeros(1,size(cardiac_window,3));
    fpower_array_channels = zeros(1,size(cardiac_window,3));
    for k = 1:size(cardiac_window,3) % Channels iteration
        %cross-correlate the two wavelength signals - both should have cardiac pulsations
        similarity = xcorr(squeeze(cardiac_window(1,:,k)),squeeze(cardiac_window(2,:,k)),'unbiased');
        if any(abs(similarity)>eps)
            % this makes the SCI=1 at lag zero when x1=x2 AND makes the power estimate independent of signal length, amplitude and Fs
            similarity = length(squeeze(cardiac_window(1,:,k)))*similarity./sqrt(sum(abs(squeeze(cardiac_window(1,:,k))).^2)*sum(abs(squeeze(cardiac_window(2,:,k))).^2));
            similarity(isnan(similarity)) = 0;
            [pxx,f] = periodogram(similarity,hamming(length(similarity)),length(similarity),fs,'power');
            [pwrest,idx] = max(pxx(f<fcut_max)); % FIX Make it age-dependent
            sci=similarity(length(squeeze(cardiac_window(1,:,k))));
            power=pwrest;
            fpower=f(idx);
            sci_array_channels(k) = sci;
            power_array_channels(k) = power;
            fpower_array_channels(k) = fpower;
        else
            warning('Similarity results close to zero');           
            sci_array_channels(k) = 0;
            power_array_channels(k) = 0;
            fpower_array_channels(k) = -1;
        end

    end
    sci_array(:,j) = sci_array_channels;    % Adjust not based on machine
    power_array(:,j) = power_array_channels;
    fpower_array(:,j) = fpower_array_channels;
end

end
