%% stats from Ashlee's paper
id_asl = 1;
sigmaGauss = 1.5; % sigma used to smooth neurons for tip detection
div = 0.5; % based on Ashlee's paper. div will be 0.5,1,1.5,2,3,4,6
% ratio to convert nm from Ashlee's paper to neuron growth pixels
image_ratio = 2.5;
segmentLength = [5.05 7.82;
    5.58 8.36;
    5.64 9.39;
    6.39 10.35;
    8.35 10.95;
    6.65 10.86;
    10.24 12.36]*image_ratio;
sigma_mu = [22.18,11.65; % absolute relative change angle
    22.82,10.82;
    21.34,9.69;
    22.32,8.01;
    22.88,6.78;
    21.28,4.91;
    20.32,2.65];
randSign = [-1,1]; % for randomly flipping signs of relative change angle
tort_quantile = [1.0225,1.0776; % tortuosity data
    1.0161,1.0757;
    1.0254,1.0507;
    1.0283,1.0685;
    1.0300,1.0725;
    1.0341,1.0623;
    1.0302,1.0498];
degrees = [1,2.5; % number of end points (degrees)
    1,3;
    1,4;
    1,4;
    2,6;
    3,7;
    6,10];
numTips = zeros(1,length(degrees));
numTips(1) = randi(floor(degrees(1,:)));
for i = 2:length(degrees)
    numTips(i) = randi(floor(degrees(i,:)));
    while numTips(i) < numTips(i-1)
        numTips(i) = randi(floor(degrees(i,:)));
    end
end
numTips
% corresponding div:      0.5,  1,    1.5,  2,    3,     4,     6.           
meanNeuriteLength = [27.53,36.54,53.19,84.34,155.13,218.74,554.73]...
    *image_ratio;
totalNeuriteLength = 0; % initialize to 0

disp('Ashlee States Parameter Initialization - Done!');