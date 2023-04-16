function [neurites] = neuritesCleanUp(neurites,threshold)

    L_neurites = bwconncomp(neurites,4);
    S_neurites = regionprops(L_neurites,'Area');
    for o=1:length(S_neurites)
        if(S_neurites(o).Area)<threshold
            neurites(cell2mat(L_neurites.PixelIdxList(o)))=0;
        end
    end

end