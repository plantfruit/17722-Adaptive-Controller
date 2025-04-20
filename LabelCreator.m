% For the ML task

pulseNum = 1; % Number of pulses extracted from each file
fileNum = 20; % Number of trials conducted for each label
gridSize = 6; % Number of labels, or points in the grid. E.g. If it's a 5x5 grid, we have 25 points

labels = zeros(gridSize * fileNum * pulseNum, 1);
%labels = strings(gridSize * fileNum * pulseNum, 1);

objNames = {"Stylus", "Screwdriver", "Battery", "Plug", "Motor", "Tripod"};

for i = 1:gridSize
    for j = 1:fileNum * pulseNum
        labels((i - 1) * fileNum * pulseNum + j) = i;% objNames{i};
    end
end

save("dir6_2_labels.txt", "labels", "-ascii")