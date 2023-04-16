% function [phi_sum] = sum_filter(phi,kernal_circle,tip_threshould,cutoff)
%     % THis function takes phi as input and output a phi_sum variable that has
%     % maximum value at tips (0~1)
% 
% %     L = bwconncomp(phi,4);
% %     ID = zeros(size(phi));
% %     for i = 1:L.NumObjects
% %         ID(L.PixelIdxList{i}) = i;
% %     end
% %     
% %     ID(L.PixelIdxList{k}) = k;
% 
%     phi_sum = conv2(phi,kernal_circle,'same').*phi;
%     
%     phi_sum(isnan(phi_sum))=0;
%     phi_sum(phi_sum>tip_threshould)=0;
%     
%     L_tip = bwconncomp(phi_sum,4);
%     S_tip = regionprops(L_tip,'Area');
%     for o=1:length(S_tip)
%         if(S_tip(o).Area)<cutoff
%             phi_sum(cell2mat(L_tip.PixelIdxList(o)))=0;
%         end
%     end
% 
% end

function [phi_sum] = sum_filter(phi,kernal_sz,tip_threshould,cutoff)
    % THis function takes phi as input and output a phi_sum variable that has
    % maximum value at tips (0~1)

%     L = bwconncomp(phi,4);
%     ID = zeros(size(phi));
%     for i = 1:L.NumObjects
%         ID(L.PixelIdxList{i}) = i;
%     end
%     
%     ID(L.PixelIdxList{k}) = k;

%     ind = 5;
%     [sz,~] = size(phi);
%     tmp = zeros(sz+ind-1);
%     tmp(ind:end,ind:end) = phi;

    phi_sum = conv2(tmp,ones(kernal_sz),'same').*tmp;
    
    phi_sum(isnan(phi_sum))=0;
    phi_sum(phi_sum>tip_threshould)=0;
    
    L_tip = bwconncomp(phi_sum,4);
    S_tip = regionprops(L_tip,'Area');
    for o=1:length(S_tip)
        if(S_tip(o).Area)<cutoff
            phi_sum(cell2mat(L_tip.PixelIdxList(o)))=0;
        end
    end

%     phi_sum = phi_sum(1:end-ind,1:end-ind);

end