close all
clear

% Tone parameters (reused from the Microphone app)
Fs = 44100;         % Oops I thought it was 48 kHz when it was actually 44 kHz, for the Voice Recorder app
fadeLength = 1;         % Reduce amplitude of signal gradually to 0, units are in seconds
gapLength = 0; %2;          % For chirps and sweeps

% Frequency of created signal
chatFreq = 659.6;
pvx1 = 691.532; pvx2 = 2074.6; pvx3 = 3457.7;
% Sweep frequencies
sweepFreq = [0 20e3]; %[18e3 20e3]; %[0 16e3];
signalLength = 600; %0.1       % In seconds
totalSignalLength = 600;        % The above one is for pulse, this is for like the entire thing.

% Edit these parameters to customize the outputted signal
fadeoutAtEnd = false;
signalType = "triangle";
frequency = chatFreq;
deltaFreq = 12000; % Alter this parameter to change the number of delta pulses in the timespan of the signal
numPulses = 8;          % Number of times the pulse is repeated (for non-delta signals)

% Build the selected signal, based on input provided in above parameter
t = 1/Fs:1/Fs:signalLength;  % Time vector for the chirp signal (generally read only now)
finalSignal = [];
switch (signalType)
    case "sawtooth"
        finalSignal = sawtooth(2*pi*frequency*t);
    case "triangle"
        finalSignal = sawtooth(2*pi*frequency*t, 0.5);
    case "square"
        finalSignal = square(2*pi*frequency*t);
    case "sweep"
        for i = 1:totalSignalLength/signalLength%numPulses
            finalSignal = [finalSignal chirp(t, sweepFreq(1), max(t), sweepFreq(2), 'linear')]; % zeros(1, Fs * gapLength)];
        end
    case "chirp"
        for i = 1:numPulses
            finalSignal = [finalSignal chirp(t, sweepFreq(1), max(t), sweepFreq(2)) zeros(1, Fs * gapLength)];
        end
    case "whitenoise"
        finalSignal = randn(1,totalSignalLength * Fs);
        finalSignal = finalSignal / max(abs(finalSignal));
    case "pinknoise"
        finalSignal = pinknoise(totalSignalLength * Fs);
        finalSignal = finalSignal / max(abs(finalSignal));
        finalSignal = finalSignal';
    case "rectpulse"
        for i = 1:numPulses
            finalSignal = [finalSignal ones(1, 2 * max(t) * Fs) zeros(1, Fs * gapLength)];
        end
    case "deltapulse"
        finalSignal = zeros(1, totalSignalLength * Fs);        deltaGap = round(length(finalSignal)/deltaFreq);
        for i = 1:deltaFreq
            finalSignal(i * deltaGap) = 1;
        end
        
        % for i = 1:numPulses
        %     %deltaFreq = 
        % 
        %     deltaSegment = zeros(1, length(t));
        %     deltaSegment(round(length(t)/2)) = 1;
        %     finalSignal = [finalSignal deltaSegment zeros(1, Fs * gapLength)];
        % end

    case "gaussianpulse"
        finalSignal = zeros(1, totalSignalLength * Fs);
        tc = gauspuls('cutoff',50e3,0.6,[],-40); 
        t = -tc : 1e-7 : tc; 
        [yi,yq,ye] = gauspuls(t,50e3,0.6); 

        deltaGap = round(length(finalSignal)/deltaFreq);
        for i = 1:deltaFreq
            finalSignal(i * deltaGap: i*deltaGap + length(ye) -1 ) = ye;
        end

        finalSignal = finalSignal / max(abs(finalSignal));
        finalSignal = finalSignal - 0.5 * max(abs(finalSignal)); % Amplify it because gaussian is soft for some reason
        finalSignal = finalSignal * 50;
    otherwise
        % Do nothing
end

% Introduce fadeout (by duplicating the first chirp)
if (fadeoutAtEnd == true)
    % Credit: Matt Mizumi on StackExchange for fadeout code
    fadeScale = linspace(1, 0, fadeLength * Fs)';
    finalSignal(end - length(fadeScale) + 1:end) = (finalSignal(end - length(fadeScale) + 1:end)) .* fadeScale'; % apply fade
end

figure; plot(finalSignal);

audiowrite(signalType + ".wav", int16(finalSignal * 32767), Fs);