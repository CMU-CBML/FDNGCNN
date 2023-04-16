function [trace_cell] = trace_neurites(neurites_dist)

    trace_cell = {};
    tc_step = 10;
    [~,~,numNeurites] = size(neurites_dist);
    for z = 1:numNeurites
        trace = [];
        tmp = neurites_dist(:,:,z);
        tmp(tmp==0) = NaN;
        max_tmp = max(max(tmp));
        min_tmp = min(min(tmp));
        
        [sy,sx] = find(tmp==max_tmp);
        start_x = sx(1); start_y = sy(1);
        trace(1,:) = [start_y,start_x];
        [last_y,last_x] = find(tmp==min_tmp);
    
        current_pVal = tmp(start_y,start_x);
        current_y = start_y;
        current_x = start_x;
        while current_pVal > tmp(last_y(1),last_x(1))
            surrounding_pVal = tmp(current_y-tc_step:current_y+tc_step,...
                current_x-tc_step:current_x+tc_step);
            surrounding_min_pVal = min(min(surrounding_pVal));
            [nxt_tc_y,nxt_tc_x] = find(surrounding_pVal==surrounding_min_pVal);
            current_y = current_y+nxt_tc_y(1)-tc_step-1;
            current_x = current_x+nxt_tc_x(1)-tc_step-1;
            trace(end+1,1:2) = [current_y,current_x];    
            current_pVal = tmp(trace(end,1),trace(end,2));
        end
        trace_cell{z} = trace;
    end
end