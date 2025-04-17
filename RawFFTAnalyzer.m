close all

% Folder Names
fftTest1 = 'Mic Data/Apr 17 Phone FFT Test';

% Select folder to analyze
folderPath = fftTest1;

% Source: https://www.mathworks.com/matlabcentral/answers/411500-how-do-i-read-all-the-files-in-a-folder
files = dir([folderPath '/*.txt']);
fileNames = files;

for k = 1:length(fileNames)
    fileName = [folderPath '/' files(k).name];

    fftData = readmatrix(fileName);

    figure
    plot(fftData)
end