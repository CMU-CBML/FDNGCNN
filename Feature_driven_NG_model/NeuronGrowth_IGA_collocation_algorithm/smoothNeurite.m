function [output] = smoothNeurite(phi_plot,sigma)
    [lenu,lenv] = size(phi_plot);
    output = imgaussfilt(full(reshape(phi_plot,lenu,lenv)),sigma);
    output(output>1e-1)=1; output(output<=1e-1)=0;
end