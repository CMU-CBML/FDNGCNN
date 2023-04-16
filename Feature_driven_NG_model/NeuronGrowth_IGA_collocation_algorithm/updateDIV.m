function [div,id_asl] = updateDIV(totalNeuriteLength,meanNeuriteLength)
    % Update div for use in next iteration (0.5,1,1.5,2,3,4,6)
    if totalNeuriteLength < meanNeuriteLength(1)
        div = 0.5; id_asl = 1;
    elseif totalNeuriteLength < meanNeuriteLength(2)
        div = 1; id_asl = 2;
    elseif totalNeuriteLength < meanNeuriteLength(3)
        div = 1.5; id_asl = 3;
    elseif totalNeuriteLength < meanNeuriteLength(4)
        div = 2; id_asl = 4;
    elseif totalNeuriteLength < meanNeuriteLength(5)
        div = 3; id_asl = 5;
    elseif totalNeuriteLength < meanNeuriteLength(6)
        div = 4; id_asl = 6;
    elseif totalNeuriteLength < meanNeuriteLength(7)
        div = 6; id_asl = 7;
    end
end

% if totalNeuriteLength < meanNeuriteLength(1)
%     div = 0.5; id_asl = 1;
% elseif totalNeuriteLength < meanNeuriteLength(2)
%     div = 1; id_asl = 2;
% elseif totalNeuriteLength < meanNeuriteLength(3)
%     div = 1.5; id_asl = 3;
% elseif totalNeuriteLength < meanNeuriteLength(4)
%     div = 2; id_asl = 4;
% elseif totalNeuriteLength < meanNeuriteLength(5)
%     div = 3; id_asl = 5;
% elseif totalNeuriteLength < meanNeuriteLength(6)
%     div = 4; id_asl = 6;
% elseif totalNeuriteLength < meanNeuriteLength(7)
%     div = 6; id_asl = 7;
% else
%     end_iter = iter;
%     fprintf(['Passed div6, ending simulation at iter:',...
%         num2str(end_iter),'\n']);
% end