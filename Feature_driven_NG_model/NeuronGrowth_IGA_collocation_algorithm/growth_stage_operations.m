phi_plot = round(full(reshape(cm.NuNv*phi,lenu,lenv)));
if iter < 1500 % transition stage
% if iter < 500 % transition stage
    % identification of all tips
%     tip = 1./sum_filter(round(phi_plot),20,175,5);
%     tip = 1./sum_filter(round(phi_plot),20,185,5);
%     tip = 1./sum_filter(round(phi_plot),kernel_circle,100,5);
%     tip(isinf(tip))=0; tip(isnan(tip))=0; 
%     tip = sum_filter_new(phi_plot,175,5);
    tip = sum_filter_old(phi_plot);
    tip(isinf(tip))=0; tip(isnan(tip))=0; 
    regionalMax = imregionalmax(tip);
    [Max_y,Max_x] = find(regionalMax);
    if iter <= 800; phi_mask = phi_plot; end

else % Growth operations based on div'
    % expand phi_mask based on current domain size (always different)
    [mask_sz,~] = size(phi_mask); phi_mask_expanded = zeros(lenu,lenv);
    sz_diff = abs(floor((lenu-mask_sz)/2));
    phi_mask_expanded(sz_diff+1:mask_sz+sz_diff,...
        sz_diff+1:mask_sz+sz_diff) = phi_mask;
    far_section = phi_plot-phi_mask_expanded;
    % identification of neurons
    L = bwconncomp(phi_plot,4);
    numPixels = cellfun(@numel,L.PixelIdxList);
    S = regionprops(L,'Centroid');
    centroids = floor(cat(1,S.Centroid));
    % loop through all neurons and adjust tips for each neuron
    Max_x = []; Max_y = [];
    far_section_all = zeros(lenu,lenv,L.NumObjects);
    dist= zeros(lenu,lenv,L.NumObjects); % geodesic dist of neurons
    neurites_x = []; neurites_y = []; M_tmp = zeros(lenu,lenv);
    numConnected = zeros(1,L.NumObjects);
    totalNeuriteLength = 0;
    for k = 1:L.NumObjects % loop through all neurons
        if numPixels(k) <= 1500
            numConnected(k) = 1;
        else
            numConnected(k) = min(floor(numPixels(k)/1500),numNeuron);
        end
        Max_xs = []; Max_ys = [];
        % calculate geodesic distance of the neuron
        ID = zeros(lenu,lenv); ID(L.PixelIdxList{k}) = 1;
        [singleNeuriteLength,tracing_tmp] = neuriteLengthFromTracing(ID);
        if k == 1; tracing = tracing_tmp;
        else tracing{end+1} = tracing_tmp{:}; end

        totalNeuriteLength = totalNeuriteLength+singleNeuriteLength;
        ID = smoothNeurite(ID,sigmaGauss);
        far_section_all(:,:,k) = reshape(far_section.*ID,lenu,lenv).';
        [nid,~] = dsearchn([seed_y+lenv/2;seed_x+lenu/2].',...
            centroids(k,:));
        dist(:,:,k) = bwdistgeodesic(logical(round(phi_plot)),...
            floor(lenv/2)+seed_y(nid),...
            floor(lenu/2)+seed_x(nid),'quasi-euclidean')-seed_radius;
        dist(isinf(dist))=0; dist(isnan(dist))=0; dist(dist<0)=0;

        % detect tips based on degrees threshold from Ashlee paper
        tip_threshould = 175; 
        tip = sum_filter_new(ID,tip_threshould,5);

        L_tip = bwconncomp(tip,8);
        % while loop to get tips based on degrees
        id_tip_loop = 0;
        
        while L_tip.NumObjects~=(numTips(id_asl)*numConnected(k))
            if L_tip.NumObjects<(numTips(id_asl)*numConnected(k))
                tip_threshould = tip_threshould + 5;
            elseif L_tip.NumObjects>=(numTips(id_asl)*numConnected(k))
                tip_threshould = tip_threshould - 5;
            end
            tip = sum_filter_new(ID,tip_threshould,5);

            L_tip = bwconncomp(tip,8);
            if (id_tip_loop>20) && (L_tip.NumObjects~=0); break; end
            id_tip_loop = id_tip_loop+1;
        end
        S_tip = regionprops(L_tip,'Centroid');
        centroids_tip = floor(cat(1,S_tip.Centroid));

        if iter == 1500 % initialize at 100 iter
            for jj = 1:100 % temporary work-around (so that there 
                % is enough changeAngle for later indexing)
                changeAngle(jj) = randSign(randi(2))*...
                    normrnd(sigma_mu(id_asl,2),sigma_mu(id_asl,1));
            end
            numCP = L_tip.NumObjects;
        end

        segLen = totalNeuriteLength/numCP;
        if ((segLen <= segmentLength(id_asl,2)) && (segLen >= segmentLength(id_asl,1))) || (iter == 1500) || mod(iter,10)==0
            if k == 1
                cx = []; cy = []; id_tips = 1; 
            end
            % loop through all tips
            for l = 1:L_tip.NumObjects
                cdd_closest = 100;
                for i=1:length(tracing)
                    for j=1:length(tracing{i})
                        trace_tmp = tracing{i}{j};
                        [cid_tmp,cdd_tmp] = dsearchn(trace_tmp,[centroids_tip(l,2),centroids_tip(l,1)]);
                        if cdd_tmp<=cdd_closest
                            trace_closest = round(trace_tmp);
                            cid_closest = cid_tmp;
                            cdd_closest = cdd_tmp;
                        end
                    end
                end
                
                if (segLen <= segmentLength(id_asl,2)) && (segLen >= segmentLength(id_asl,1))
                    numCP = numCP + 1;
                end

                neurites_dist = dist(:,:,k);
                closest_id = trace_closest(cid_closest,:);
                beginning_id = trace_closest(end,:);
                furthest_id = trace_closest(1,:);
                dist_closet = max(max(neurites_dist(closest_id(1)-5:closest_id(1)+5,closest_id(2)-5:closest_id(2)+5)));
                dist_beginning = min(min(neurites_dist(beginning_id(1)-5:beginning_id(1)+5,beginning_id(2)-5:beginning_id(2)+5)));
                neurite_geodesic_dist = dist_closet - dist_beginning;
                dist_furthest = max(max(neurites_dist(furthest_id(1)-5:furthest_id(1)+5,furthest_id(2)-5:furthest_id(2)+5)));

                tort = 10; tort_idx = 0;
                while max(max(exist('tort','var')))==0 || ...
                    max(max(tort))>tort_quantile(id_asl,2)||...
                    min(min(tort))<tort_quantile(id_asl,1)
                    
                    offsetAngle = randSign(randi(2))*rand*90;
                    if neurites_dist(centroids_tip(l,2),centroids_tip(l,1)) < (dist_furthest-5)                        
                        changeAngle(l) = randSign(randi(2))*...
                            (normrnd(sigma_mu(id_asl,2),sigma_mu(id_asl,1))+offsetAngle);
                    else
                        changeAngle(l) = randSign(randi(2))*...
                            normrnd(sigma_mu(id_asl,2),sigma_mu(id_asl,1));
                    end

                    [cy_tmp,cx_tmp] = expDist(beginning_id(1),beginning_id(2),...
                        closest_id(1),closest_id(2),changeAngle(l));
                    dist2center = sqrt((cx_tmp-beginning_id(1)).^2+(cy_tmp-beginning_id(2)).^2);
                    distTipExtend = sqrt((cx_tmp-closest_id(1)).^2+(cy_tmp-closest_id(2)).^2);
                    tort = (neurite_geodesic_dist+distTipExtend)./dist2center;
                
                    % stop when tort cant recover
                    if tort_idx>20; break; end
                    tort_idx = tort_idx+1;
                end
                cx(id_tips) = cx_tmp; cy(id_tips) = cy_tmp;
                id_tips = id_tips + 1;
            end % end tips loop
        end
        % calculating Max_x/y of E zones based on cue guiding
        numCues = length(cx);
        cue = zeros(lenu,lenu,numCues);
        for z=1:numCues % constructing a cue field for each external cue
            cue(:,:,z) = 1./sqrt(bsxfun(@times,(X-cx(z)),(X-cx(z)))+...
                bsxfun(@times,(Y-cy(z)),(Y-cy(z)))).';
            cue(isinf(cue))=1; cue(isnan(cue))=1;
            cue_on_far = cue(:,:,z).*abs(far_section_all(:,:,k)).';
            [max_ys, max_xs] = find(cue_on_far==...
                max(max(cue_on_far))); % selecting tips based on cue
            Max_xs(z) = max_xs(1);
            Max_ys(z) = max_ys(1);
        end
        Max_x = [Max_x,Max_xs]; Max_y = [Max_y,Max_ys];
        if isempty(Max_xs) ~= 1
            % calculate furthest tip and adjust M_phi accordingly
            Max_s = zeros(length(Max_xs),3);
            for zz = 1:length(Max_xs)
                Max_s(zz,1) = Max_xs(zz);
                Max_s(zz,2) = Max_ys(zz);
                Max_s(zz,3) = dist(Max_ys(zz),Max_xs(zz),k);
            end
            [~,idx] = sort(Max_s(:,3)); % sort just the first column
            sortedmat = Max_s(idx,:); 
            mx = sortedmat(end,1);
            my = sortedmat(end,2);
            theta_ori_phi = highlightZone(lenu,lenv,mx,my,length(mx),...
                gc_sz);
        else
            theta_ori_phi = zeros(lenu,lenv);
        end
        M_tmp = M_tmp+theta_ori_phi;
    end % end neurons loop
    M_tmp(M_tmp~=0) = M_axon; M_tmp(M_tmp==0) = M_neurites;
    M_phi = reshape(M_tmp,lenu*lenv,1);
    if mod(iter,100)==0
        % Update div for use in next iteration (0.5,1,1.5,2,3,4,6)
        perNeuronNeuriteLength = sum(totalNeuriteLength)/numNeuron;
        if perNeuronNeuriteLength < meanNeuriteLength(7)
            [div,id_asl] = updateDIV(perNeuronNeuriteLength,...
                meanNeuriteLength);
            if div~=prevDiv
                divLog = [divLog;div,iter];
                writematrix(divLog,'./data/divLog.txt');
            end
            prevDiv = div;
        else
            end_iter = iter;
            fprintf(['Passed div6, ending simulation at iter:',...
                num2str(end_iter),'\n']);
        end
    end
    % plotting
    if(mod(iter,png_plot_invl) == 0)
        subplot(2,3,3);
        imagesc(reshape(M_phi,lenu,lenv)); axis square; colorbar;
        title(sprintf('M_phi at iteration = %.2d',iter));
        subplot(2,3,4);
        imagesc(phi_plot+sum(cue,3)); axis square; colorbar; hold on;
        scatter(Max_x,Max_y,'cx'); hold on;
        scatter(cx,cy,'ro'); hold off
        
        title(['div:',num2str(div),' degrees:',num2str(L_tip.NumObjects)]);
        subplot(2,3,6);
        imagesc(sum(dist,3)); axis square; colorbar; hold on;
        title(['Nlength:',num2str(perNeuronNeuriteLength),...
            'segLen',num2str(segLen)]);
        for m = 1:length(tracing)
            for n = 1:length(tracing{m})
                hold on;
                plot(tracing{m}{n}(:,2),tracing{m}{n}(:,1),'Color','r');
            end
        end
        drawnow;
    end
end

% construct E zones based on Max_x/y
[theta_ori] = highlightZone(lenu,lenv,Max_x,Max_y,length(Max_x),gc_sz);
if iter >= 1500
    theta_ori = theta_ori+theta_ori_phi;
    theta_ori(theta_ori>0) = 1;
end

%% all;
% 
% tip_threshould = 175;
% % tip1 = sum_filter_new(full(phi_plot),0.7,5);
% tip2 = sum_filter_new(ID,tip_threshould,5);
% tip3 = sum_filter(ID,20,tip_threshould,5);
% 
% figure;
% subplot(1,2,1);
% imagesc(tip2); colorbar;
% subplot(1,2,2);
% imagesc(tip3); colorbar;
% 
% %%
% close all;
% 
% pp = reshape(full(phi_plot),103,103);
% tip_threshould = 185;
% tip3 = sum_filter(pp,20,tip_threshould,5);
% 
% figure;
% subplot(1,2,1);
% imagesc(tip3); colorbar;
% subplot(1,2,2);
% imagesc(full(phi_plot)); colorbar;
% 
% %%
% ppp = conv2(pp,ones(20),'full');
% size(pp)
% size(ppp)
% close