function [out,seed_x,seed_y] = expandMat(input,edgeID,expd_sz,seed_x,seed_y)
    if edgeID(1) == 1 % Top
        in_sz = length(input);
        out = zeros(length(input)+expd_sz);
        out(expd_sz+1:end,expd_sz/2+1:expd_sz/2+in_sz) = input;
        seed_x = seed_x+expd_sz/2;
%         seed_y = seed_y+expd_sz;
        input = out;
    end
    if edgeID(2) == 1 % Left
        in_sz = length(input);
        out = zeros(length(input)+expd_sz);
        out(expd_sz/2+1:expd_sz/2+in_sz,expd_sz+1:end) = input;
        seed_y = seed_y+expd_sz/2;
%         seed_x = seed_x+expd_sz;
        input = out;
    end
    if edgeID(3) == 1 % Bottom
        in_sz = length(input);
        out = zeros(length(input)+expd_sz);
        out(1:end-expd_sz,expd_sz/2+1:expd_sz/2+in_sz) = input;
        seed_x = seed_x-expd_sz/2;
%         seed_y = seed_y+expd_sz;
        input = out;
    end
    if edgeID(4) == 1 % Right
        in_sz = length(input);
        out = zeros(length(input)+expd_sz);
        out(expd_sz/2+1:expd_sz/2+in_sz,1:end-expd_sz) = input;
        seed_y = seed_y-expd_sz/2;
%         seed_x = seed_x+expd_sz;
    end
end