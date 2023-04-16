function [trace_cell,trace_length] = traceNeurites(neurites_dist,tc_step)

    trace_cell = {};
    [~,~,numNeurites] = size(neurites_dist);
    trace_length = 0;

    for z = 1:numNeurites
        trace = [];
        tmp = neurites_dist(:,:,z);
        tmp(tmp==0) = NaN;
        max_tmp = max(max(tmp));
        min_tmp = min(min(tmp));
    
        [sy,sx] = find(tmp==max_tmp);
        try
            start_x = sx(1); start_y = sy(1);
        catch
            continue
        end

        trace(1,:) = [start_y,start_x];
        [ly,lx] = find(tmp==min_tmp);
        last_x = lx(1); last_y = ly(1);
         
        current_pVal = tmp(start_y,start_x);
        current_y = start_y;
        current_x = start_x;
        prev_val = 0; loop_ind = 0;
        while current_pVal > tmp(last_y(1),last_x(1))
            surrounding_pVal = tmp(current_y-tc_step:current_y+tc_step,...
                current_x-tc_step:current_x+tc_step);
            surrounding_min_pVal = min(min(surrounding_pVal));
            [nxt_tc_y,nxt_tc_x] = find(surrounding_pVal==surrounding_min_pVal);
            current_y = current_y+nxt_tc_y(1)-tc_step-1;
            current_x = current_x+nxt_tc_x(1)-tc_step-1;
            trace(end+1,1:2) = [current_y,current_x];    
            current_pVal = tmp(trace(end,1),trace(end,2));
            if current_pVal == prev_val
                loop_ind = loop_ind + 1;
            end
            if loop_ind > 10
                break;
            end
            prev_val = current_pVal;
        end
        if length(trace) >= 5
            trace_cell{end+1} = trace;
            trace_length = trace_length ...
                + neurites_dist(trace(1,1),trace(1,2),z) ...
                - neurites_dist(trace(end,1),trace(end,2),z);
        end
    end
end