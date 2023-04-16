function [totalNeuriteLength,traces] = neuriteLengthFromTracing(phi_plot)

    up_scale = 1;
%     phi_plot_trace = round(imresize(phi_plot,up_scale,"bicubic"));
    phi_plot_trace = phi_plot;

    %%
    seed_radius = 20*up_scale;
    [phi_sz,~] = size(phi_plot_trace);
    phi_initial = conv2(phi_plot_trace,ones(30*up_scale),'same').*phi_plot_trace;
    phi_initial(isnan(phi_initial))=0;
    phi_initial(phi_initial<0.45*max(max(phi_initial))) = 0;
    neurites = phi_plot_trace - phi_initial;
    neurites(neurites<0) = 0;
    neurites(neurites~=0) = 1;

%     figure;
%     subplot(1,2,1);
%     imagesc(phi_plot);
%     subplot(1,2,2);
%     imagesc(neurites);

    L_neurites = bwconncomp(neurites,4);
    S_neurites = regionprops(L_neurites,'Area');
    for o=1:length(S_neurites)
        if(S_neurites(o).Area)<10
            neurites(cell2mat(L_neurites.PixelIdxList(o)))=0;
        end
    end
    
    L_phi_initial = bwconncomp(phi_initial,4);
    S_phi_initial = regionprops(L_phi_initial,'Centroid');
    centroids_initial = floor(cat(1,S_phi_initial.Centroid));

    %%
    dist = bwdistgeodesic(logical(round(phi_plot_trace)),...
        round(centroids_initial(1,1)),round(centroids_initial(1,2)),'quasi-euclidean')-seed_radius-5;

    dist(isinf(dist))=0; dist(isnan(dist))=0; 
    dist(dist<0)=0;

%     figure;
%     subplot(1,2,1);
%     imagesc(neurites); hold on;
%     scatter(round(phi_sz/2),round(phi_sz/2));
%     subplot(1,2,2);
%     imagesc(dist);
%     colorbar;

%     figure('Position',[50,50,800,800],'visible','off');
%     figure('Position',[50,50,800,800]);
%     imagesc(dist);
%     colormap('gray');
%     axis square; colorbar;
    
    %% primary
    [neurites] = neuritesCleanUp(neurites,10);
    L_neurites = bwconncomp(neurites,4);
    
    neurites_dist = zeros(phi_sz*phi_sz,L_neurites.NumObjects);
    for z = 1:L_neurites.NumObjects
        neurites_dist(L_neurites.PixelIdxList{z},z) = dist(L_neurites.PixelIdxList{z});
    end
    neurites_dist = reshape(neurites_dist,[phi_sz,phi_sz,L_neurites.NumObjects]);

    [trace_cell_primary,trace_length_primary] = traceNeurites(neurites_dist,2);
% 
%     for i = 1:length(trace_cell_primary)
%         hold on;
%         plot(trace_cell_primary{i}(:,2),trace_cell_primary{i}(:,1),'LineWidth',2,'Color','r');
%     end
    
    %% secondary
    phi_secondary = neurites;
    rm_sz = 4;
    for m = 1:length(trace_cell_primary)
        trace = trace_cell_primary{m};
        for n = 1:length(trace)
            i = trace(n,1); j = trace(n,2);
            phi_secondary(i-rm_sz:i+rm_sz,j-rm_sz:j+rm_sz) = 0;
        end
    end
    
    [phi_secondary] = neuritesCleanUp(phi_secondary,10);
    L_neurites_secondary = bwconncomp(phi_secondary,4);
    
    neurites_dist = zeros(phi_sz*phi_sz,L_neurites_secondary.NumObjects);
    for z = 1:L_neurites_secondary.NumObjects
        neurites_dist(L_neurites_secondary.PixelIdxList{z},z) = dist(L_neurites_secondary.PixelIdxList{z});
    end
    neurites_dist = reshape(neurites_dist,[phi_sz,phi_sz,L_neurites_secondary.NumObjects]);
    
    [trace_cell_secondary,trace_length_secondary] = traceNeurites(neurites_dist,4);
% 
%     for i = 1:length(trace_cell_secondary)
%         hold on;
%         plot(trace_cell_secondary{i}(:,2),trace_cell_secondary{i}(:,1),'LineWidth',2,'Color','c');
%     end

%%  tertiary
    phi_tertiary = phi_secondary;
    rm_sz = 4;
    for m = 1:length(trace_cell_secondary)
        trace = trace_cell_secondary{m};
        for n = 1:length(trace)
            i = trace(n,1); j = trace(n,2);
            phi_tertiary(i-rm_sz:i+rm_sz,j-rm_sz:j+rm_sz) = 0;
        end
    end
    
    [phi_tertiary] = neuritesCleanUp(phi_tertiary,10);
    L_neurites_tertiary = bwconncomp(phi_tertiary,4);
    
    neurites_dist = zeros(phi_sz*phi_sz,L_neurites_tertiary.NumObjects);
    for z = 1:L_neurites_tertiary.NumObjects
        neurites_dist(L_neurites_tertiary.PixelIdxList{z},z) = dist(L_neurites_tertiary.PixelIdxList{z});
    end
    neurites_dist = reshape(neurites_dist,[phi_sz,phi_sz,L_neurites_tertiary.NumObjects]);
    
    [trace_cell_tertiary,trace_length_tertiary] = traceNeurites(neurites_dist,4);
% 
%     for i = 1:length(trace_cell_tertiary)
%         hold on;
%         plot(trace_cell_tertiary{i}(:,2),trace_cell_tertiary{i}(:,1),'LineWidth',2,'Color','m');
%     end
    
    totalNeuriteLength = trace_length_primary+trace_length_secondary+trace_length_tertiary;

    totalNeuriteLength = totalNeuriteLength/up_scale;
%     title(strcat('Total length:', num2str(totalNeuriteLength)));
%     saveas(gcf,strcat('./output/',erase(phi_plots(p).name,'_phi_35000.mat'),'.png'));
    traces = {trace_cell_primary,trace_cell_secondary,trace_cell_tertiary};
end