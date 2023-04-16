function [output] = quantileGen(mean,minVal,maxVal,first_quartile,third_quartile)
    
    % mean = 1.0554;
    % std = 0.0446;
    % minVal = 0;
    % maxVal = 1.2;
    % first_quartile = 1.0225;
    % third_quartile = 1.0776;
    
    output = rand(100,1);
    output(1:25) = output(1:25)*(first_quartile-minVal)+minVal;
    output(25:50) = output(25:50)*(mean-first_quartile)+first_quartile;
    output(50:75) = output(50:75)*(third_quartile-mean)+mean;
    output(75:100) = output(75:100)*(maxVal-third_quartile)+third_quartile;
    
%     quantile(randNum,[0.25,0.75],"all")
%     
%     figure;
%     histfit(randNum);
end

%%
% tortuosity_mean = 1.0554;
% tortuosity_minVal = 0;
% tortuosity_maxVal = 1.2;
% tortuosity_first_quartile = 1.0225;
% tortuosity_third_quartile = 1.0776;
% 
% tortuosity_distribution = quantileGen(tortuosity_mean,tortuosity_minVal,...
%     tortuosity_maxVal,tortuosity_first_quartile,tortuosity_third_quartile);
% 
% tortuosity = randsample(tortuosity_distribution,1);