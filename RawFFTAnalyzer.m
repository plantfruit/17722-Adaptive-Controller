close all

% Setup
%========================================================================

% Folder Names
fftTest1 = 'Mic Data/Apr 17 Phone FFT Test';
fftTest2 = 'Mic Data/Apr 17 Windowed FFT Test';
d5_1 = 'Mic Data/Apr 17 D Pad 5';
d5_3 = 'Mic Data/Apr 18 Triangle';

% Constants
Fs = 48e3;

% Parameters
fftWindow = [48 150]; %[81 1601];
fftLen = 152;
subDims = [5 10];

% Switches
doPlotting = true;
doSubplots = true;
showWindowed = false;

% Select folder to analyze
folderPath = d5_3;

% Source: https://www.mathworks.com/matlabcentral/answers/411500-how-do-i-read-all-the-files-in-a-folder
files = dir([folderPath '/*.txt']);
fileNames = files;

% Processing
%========================================================================

allFFTs = zeros(length(fileNames), 152);

% Read every file in the folder
for k = 1:length(fileNames)
    fileName = [folderPath '/' files(k).name];

    % y-vector
    fftData = readmatrix(fileName);

    % fftData = smooth(fftData, 3);

    allFFTs(k,:) = fftData;

    % x-vector
    f = linspace(0,Fs, length(fftData));

    if (doPlotting)
        if (doSubplots)
            subplot(subDims(1), subDims(2), k)
        else
            figure
        end
        
        if (showWindowed) 
            windowedFFT = fftData((fftWindow(1):fftWindow(2)));
            plot(f(fftWindow(1):fftWindow(2)), windowedFFT);
        else
            plot(fftData)
        end
    end
    % plot(f, fftData)

    % subplot(1,2,1)
    % plot(f, fftData)
    % windowedFFT = fftData((fftWindow(1):fftWindow(2)));
    % subplot(1,2,2)
    % plot(f(fftWindow(1):fftWindow(2)), windowedFFT);
end