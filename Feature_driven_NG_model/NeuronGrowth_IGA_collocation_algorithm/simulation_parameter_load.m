%% loading previous results, restarting simulation
inputFiles = dir('./data/workspace*.mat');
fileNames = {inputFiles.name};
split1 = regexp(fileNames, '_', 'split');
comps = cellfun(@(C) C{2}, split1, 'Uniform',0);
split2 = regexp(comps, '.mat', 'split');% to remove mat extension
for i = 1:length(split2)
    fileIndList(i) = str2double(cell2mat(split2{1,i}));
end
[iter,fileInd] = max(fileIndList);
load(strcat('./data/',fileNames{fileInd}));

disp('Finished loading files, continue simulation.\n');
disp('********************************************************************');