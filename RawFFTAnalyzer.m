close all

% Setup
%========================================================================

% Folder Names
fftTest1 = 'Mic Data/Apr 17 Phone FFT Test';
fftTest2 = 'Mic Data/Apr 17 Windowed FFT Test';
d5_1 = 'Mic Data/Apr 17 D Pad 5';

% Constants
Fs = 48e3;

% Parameters
fftWindow = [81 1601];
fftLen = 152;

% Switches
doPlotting = false;

% Select folder to analyze
folderPath = d5_1;

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

    allFFTs(k,:) = fftData;

    % x-vector
    f = linspace(0,Fs, length(fftData));

    if (doPlotting)
        figure
        plot(fftData)
    end
    % plot(f, fftData)

    % subplot(1,2,1)
    % plot(f, fftData)
    % windowedFFT = fftData((fftWindow(1):fftWindow(2)));
    % subplot(1,2,2)
    % plot(f(fftWindow(1):fftWindow(2)), windowedFFT);
end