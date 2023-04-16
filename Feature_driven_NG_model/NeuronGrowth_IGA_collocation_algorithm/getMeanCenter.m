function [centerX, centerY] = getMeanCenter(phi_tmp)
    phi_tmp = conv2(phi_tmp,ones(20),'same').*phi_tmp;
    phi_tmp(isnan(phi_tmp))=0;
    phi_tmp(phi_tmp>200)=1;
    phi_tmp(phi_tmp~=1)=0;
    L = bwconncomp(phi_tmp,4);
    A = regionprops(L,'Area');
    for l = 1:L.NumObjects
        if A(l).Area<300
            phi_tmp(cell2mat(L.PixelIdxList(l))) = 0;
        end
    end
    L_processed = bwconncomp(phi_tmp,4);
    S = regionprops(L_processed,'Centroid');
    centerX = []; centerY = [];
    for l = 1:L_processed.NumObjects
        centerX(end+1) = S(l).Centroid(1);
        centerY(end+1) = S(l).Centroid(2);
    end
    centerX = mean(centerX);
    centerY = mean(centerY);
end