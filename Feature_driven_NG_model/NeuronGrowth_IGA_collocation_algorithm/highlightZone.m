function [theta] = highlightZone(lenu,lenv,Max_x,Max_y,size_Max,gc_sz)
    theta = zeros(lenu,lenv);
    for l=1:size_Max
        max_x = Max_x(l);
        max_y = Max_y(l);
        if max_x>gc_sz && max_x<(lenu-gc_sz) && max_y>gc_sz && max_y<(lenv-gc_sz)
            theta(max_y-gc_sz:max_y+gc_sz,max_x-gc_sz:max_x+gc_sz) = ones(gc_sz*2+1);
        end
    end
end