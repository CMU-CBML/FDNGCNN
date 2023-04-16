function [phi_sum] = sum_filter_new(phi,tip_threshold,cutoff)
    % THis function takes phi as input and output a phi_sum variable that has
    % maximum value at tips (0~1)
    
    phi = full(phi);
    % get size of input
    [Nx,Ny] = size(phi);
    % round phi -> discrete
    phi = round(phi);
    % initialize
    phi_sum = zeros(Nx,Ny);
    
    % phi = round(phi_plot);
    L = bwconncomp(phi,4);
    ID = zeros(size(phi));
    for i = 1:L.NumObjects
        ID(L.PixelIdxList{i}) = i;
    end
    
    % loop through and calculate sum of phi values around i,j
    for i = 10:Nx-9
        for k = 10:Ny-9
            baseState = ID(i,k);
            for j = k-9:k+9
                s_9 = isequal(baseState,ID(i-9,j));
                s_8 = isequal(baseState,ID(i-8,j));
                s_7 = isequal(baseState,ID(i-7,j));
                s_6 = isequal(baseState,ID(i-6,j));
                s_5 = isequal(baseState,ID(i-5,j));
                s_4 = isequal(baseState,ID(i-4,j));
                s_3 = isequal(baseState,ID(i-3,j));
                s_2 = isequal(baseState,ID(i-2,j));
                s_1 = isequal(baseState,ID(i-1,j));
                s_0 = isequal(baseState,ID(i-0,j));
                s1_ = isequal(baseState,ID(i+1,j));
                s2_ = isequal(baseState,ID(i+2,j));
                s3_ = isequal(baseState,ID(i+3,j));
                s4_ = isequal(baseState,ID(i+4,j));
                s5_ = isequal(baseState,ID(i+5,j));
                s6_ = isequal(baseState,ID(i+6,j));
                s7_ = isequal(baseState,ID(i+7,j));
                s8_ = isequal(baseState,ID(i+8,j));
                s9_ = isequal(baseState,ID(i+9,j));
    
                phi_sum(i,k) = phi_sum(i,k) + sum(s_9*phi(i-9,j) + ...
                    s_8*phi(i-8,j) + s_7*phi(i-7,j) + s_6*phi(i-6,j) + ...
                    s_5*phi(i-5,j) + s_4*phi(i-4,j) + s_3*phi(i-3,j) + ...
                    s_2*phi(i-2,j) + s_1*phi(i-1,j) + s_0*phi(i,j) + ...
                    s1_*phi(i+1,j) + s2_*phi(i+2,j) + s3_*phi(i+3,j) + ...
                    s4_*phi(i+4,j) + s5_*phi(i+5,j) + s6_*phi(i+6,j) + ...
                    s7_*phi(i+7,j) + s8_*phi(i+8,j) + s9_*phi(i+9,j));
            end
        end
    end
    
%     figure;
%     subplot(1,2,1);
%     imagesc(phi);
%     subplot(1,2,2);
%     imagesc(phi_sum);

    phi_sum(isnan(phi_sum))=0;
    phi_sum(phi_sum>tip_threshold)=0;
%     
    L_tip = bwconncomp(phi_sum,4);
    S_tip = regionprops(L_tip,'Area');
    for o=1:length(S_tip)
        if(S_tip(o).Area)<cutoff
            phi_sum(cell2mat(L_tip.PixelIdxList(o)))=0;
        end
    end
    
end