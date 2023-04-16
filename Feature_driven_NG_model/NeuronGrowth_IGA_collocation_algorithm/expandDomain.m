function [phi_out,phi_mask_out,theta_ori_phi_out,M_phi_out,theta_out,conc_t_out,tempr_out,LAPpk_out,...
    phi_initial_out,theta_initial_out,tempr_initial_out,bcid,seed_x,...
    seed_y,changePoints] = expandDomain(phi,phi_mask,theta_ori_phi,M_phi,theta,conc_t,tempr,LAPpk,oldNuNv,...
    edgeID,expd_sz,seed_x,seed_y,changePoints)

M = sqrt(length(phi));

% expand phi mat
[phi_out,seed_x,seed_y] = expandMat(reshape(phi,M,M),edgeID,expd_sz,...
    seed_x,seed_y);
out_sz = length(phi_out);

% expand phi_mask mat
[phi_mask_out,~,~] = expandMat(reshape(phi_mask,M,M),edgeID,...
    expd_sz,seed_x,seed_y);

% expand theta_ori_phi mat
[theta_ori_phi_out,changePoints(:,1),changePoints(:,2)] = expandMat(reshape(theta_ori_phi,M,M),edgeID,...
    expd_sz,changePoints(:,1),changePoints(:,2));

% expand M_phi mat
[M_phi_out,~,~] = expandMat(reshape(M_phi,M,M),edgeID,expd_sz,seed_x,...
    seed_y);

% expand theta mat
[theta_out,~,~] = expandMat(reshape(oldNuNv*theta,M,M),edgeID,expd_sz,...
    seed_x,seed_y);
tmp = rand(out_sz);
theta_out(theta_out==0) = tmp(theta_out==0);

% expand conc_t mat
[conc_t_out,~,~] = expandMat(reshape(oldNuNv*conc_t,M,M),edgeID,expd_sz,...
    seed_x,seed_y);

% expand tempr mat
[tempr_out,~,~] = expandMat(reshape(tempr,M,M),edgeID,expd_sz,seed_x,seed_y);

% expand LAPpk mat
[LAPpk_out,~,~] = expandMat(reshape(LAPpk,M,M),edgeID,expd_sz,seed_x,seed_y);

% expand initial condition mat
phi_initial_out = reshape(phi_out,out_sz,out_sz);
theta_initial_out = reshape(theta_out,out_sz,out_sz);
tempr_initial_out = reshape(tempr_out,out_sz,out_sz);
for i = 2:out_sz-1
    for j = 2:out_sz-1
        phi_initial_out(i,j) = 0;
        theta_initial_out(i,j) = 0;
        tempr_initial_out(i,j) = 0;
    end
end
phi_initial_out = reshape(phi_initial_out,out_sz*out_sz,1);
theta_initial_out  = reshape(theta_initial_out,out_sz*out_sz,1);
tempr_initial_out  = reshape(tempr_initial_out,out_sz*out_sz,1);

phi_out = sparse(reshape(phi_out,out_sz^2,1));
conc_t_out = sparse(reshape(conc_t_out,out_sz^2,1));
tempr_out = sparse(reshape(tempr_out,out_sz^2,1));
LAPpk_out = sparse(reshape(LAPpk_out,out_sz^2,1));

% expand boundary id mat
bcid = zeros([out_sz,out_sz]);
for i = 1:out_sz
    bcid(1,i) = 1;
    bcid(end,i) = 1;
    bcid(i,1) = 1;
    bcid(i,end) = 1;
end
bcid = reshape(bcid,out_sz^2,1);
bcid = sparse(bcid);

%% Backup code
% function [phi_out,phi_mask_out,M_phi_out,theta_out,conc_t_out,tempr_out,LAPpk_out,phi_initial_out,theta_initial_out,tempr_initial_out,bcid] = ...
%     expandDomain(sz,phi,phi_mask,M_phi,theta,conc_t,tempr,LAPpk,oldNuNv,newNuNv)
% 
% len = length(phi);
% M = sqrt(len);
% 
% phi = reshape(phi,M,M);
% phi_mask = reshape(phi_mask,M,M);
% M_phi = reshape(M_phi,M,M);
% % theta_ori = reshape(theta_ori,M,M);
% theta = reshape(oldNuNv*theta,M,M);
% conc_t = reshape(oldNuNv*conc_t,M,M);
% LAPpk = reshape(LAPpk,M,M);
% tempr = reshape(tempr,M,M);
% 
% % % not necessary, just to be safe (stiffness setup insurance)
% % bcs = 4;
% % phi = phi(bcs:end-bcs,bcs:end-bcs);
% % theta = theta(bcs:end-bcs,bcs:end-bcs);
% % conc_t = conc_t(bcs:end-bcs,bcs:end-bcs);
% % tempr = tempr(bcs:end-bcs,bcs:end-bcs);
% % LAPpk = LAPpk(bcs:end-bcs,bcs:end-bcs);
% 
% [M, ~] = size(phi);
% out_sz = sqrt(sz);
% 
% phi_out =zeros(out_sz);
% phi_mask_out =zeros(out_sz);
% M_phi_out =ones(out_sz)*min(min(M_phi));
% % theta_ori_out =zeros(out_sz);
% theta_out =rand(out_sz);
% conc_t_out = zeros(out_sz);
% tempr_out = zeros(out_sz);
% LAPpk_out = zeros(out_sz);
% 
% i_off = floor(out_sz/2-M/2);
% j_off = floor(out_sz/2-M/2);
% 
% phi_out(3+i_off:M-2+i_off,3+j_off:M-2+j_off) = phi(3:M-2,3:M-2);
% phi_mask_out(3+i_off:M-2+i_off,3+j_off:M-2+j_off) = phi_mask(3:M-2,3:M-2);
% M_phi_out(3+i_off:M-2+i_off,3+j_off:M-2+j_off) = M_phi(3:M-2,3:M-2);
% % theta_ori_out(3+i_off:M-2+i_off,3+j_off:M-2+j_off) = theta_ori(3:M-2,3:M-2);
% theta_out(3+i_off:M-2+i_off,3+j_off:M-2+j_off) = theta(3:M-2,3:M-2);
% conc_t_out(3+i_off:M-2+i_off,3+j_off:M-2+j_off) = conc_t(3:M-2,3:M-2);
% tempr_out(3+i_off:M-2+i_off,3+j_off:M-2+j_off) = tempr(3:M-2,3:M-2);
% LAPpk_out(3+i_off:M-2+i_off,3+j_off:M-2+j_off) = LAPpk(3:M-2,3:M-2);
% 
% phi_initial_out = reshape(phi_out,out_sz,out_sz);
% theta_initial_out = reshape(theta_out,out_sz,out_sz);
% tempr_initial_out = reshape(tempr_out,out_sz,out_sz);
% for i = 2:out_sz-1
%     for j = 2:out_sz-1
%         phi_initial_out(i,j) = 0;
%         theta_initial_out(i,j) = 0;
%         tempr_initial_out(i,j) = 0;
%     end
% end
% phi_initial_out = reshape(phi_initial_out,out_sz*out_sz,1);
% theta_initial_out  = reshape(theta_initial_out,out_sz*out_sz,1);
% tempr_initial_out  = reshape(tempr_initial_out,out_sz*out_sz,1);
% 
% phi_out =sparse(reshape(phi_out,out_sz^2,1));
% theta_out =sparse(newNuNv\reshape(theta_out,out_sz^2,1));
% conc_t_out =sparse(newNuNv\reshape(conc_t_out,out_sz^2,1));
% tempr_out =sparse(reshape(tempr_out,out_sz^2,1));
% LAPpk_out =sparse(reshape(LAPpk_out,out_sz^2,1));
% 
% bcid = zeros([out_sz,out_sz]);
% for i = 1:out_sz
%     bcid(1,i) = 1;
%     bcid(end,i) = 1;
%     bcid(i,1) = 1;
%     bcid(i,end) = 1;
% end
% bcid = reshape(bcid,out_sz^2,1);
% bcid = sparse(bcid);
