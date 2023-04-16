function [phi,conct,seed_x,seed_y] = initialize_neurite_growth_rand2(seed_radius,lenu,lenv,numNeuron)
% this function is was initially written for sparse matrix. The main.m code
% was later changed to full mat.
seed = (seed_radius)^2;

ind_i = [];
ind_j = [];
phi_val = [];
conct_val = [];

phi = 2;
while (max(phi)>1)
    ind_i = [];
    ind_j = [];
    phi_val = [];
    conct_val = [];   

%     if numNeuron == 1 % 1 neuron
%         seed_x = [0];
%         seed_y = [0];
%     elseif numNeuron == 2 % 2 neurons
%         seed_x = [-35,35];
%         seed_y = [-35,35];
%     elseif numNeuron == 3 % 3 neurons
%         seed_x = [-45,45,45];
%         seed_y = [0,45,-45];
%     elseif numNeuron == 4 % 4 neurons
%         seed_x = [-45,45,45,-45];
%         seed_y = [-45,45,-45,45];
%     elseif numNeuron == 5 % 5 neurons
%         seed_x = [-60,60,60,-60,0];
%         seed_y = [-60,60,-60,60,0];
%     elseif numNeuron == 6 % 6 neurons
%         seed_x = [-70,-20,70,-70,20,70];
%         seed_y = [-20,-70,-70,70,70,20];
%     elseif numNeuron == 7 % 7 neurons
%         seed_x = [-75,-55,55,-55,55,75,0];
%         seed_y = [0,-75,-75,75,75,0,0];
%     end
    
    if numNeuron == 1 % 1 neuron
        seed_x = [0];
        seed_y = [0];
    elseif numNeuron == 2 % 2 neurons
        seed_x = [45,-45];
        seed_y = [45,-45];
    elseif numNeuron == 3 % 3 neurons
        seed_x = [-60,60,60];
        seed_y = [0,40,-40];
    elseif numNeuron == 4 % 4 neurons
        seed_x = [-60,60,60,-60];
        seed_y = [-60,60,-60,60];
    elseif numNeuron == 5 % 5 neurons
        seed_x = [-75,75,75,-75,0];
        seed_y = [-75,75,-75,75,0];
    elseif numNeuron == 6 % 6 neurons
        seed_x = [-85,-10,85,-85,10,85];
        seed_y = [-15,-85,-85,85,85,15];
    elseif numNeuron == 7 % 7 neurons
        seed_x = [-95,-45,45,-45,45,95,0];
        seed_y = [0,-95,-95,95,95,0,0];
        end

    for i=1:lenu
        for j=1:lenv
            if ((i-lenu/2)*(i-lenu/2)+(j-lenv/2)*(j-lenv/2) < seed)
                r = sqrt((i-lenu/2)*(i-lenu/2)+(j-lenv/2)*(j-lenv/2));
                for l = 1:numNeuron
                    ind_i(end+1) = i+seed_x(l);
                    ind_j(end+1) = j+seed_y(l);
                    phi_val(end+1) = 1;   
                    conct_val(end+1) = (0.5+0.5*tanh((sqrt(seed)-r)/2));
                end
            end
        end
    end
    %% Creating sparse matrix
    phi = sparse(ind_i,ind_j,phi_val,lenu,lenv);
    phi = full(reshape(phi,lenu*lenv,1)); % reshpae phi for calculation
    conct = sparse(ind_i,ind_j,conct_val,lenu,lenv);
    conct = full(reshape(phi,lenu*lenv,1)); % reshpae phi for calculation

end

%     if numNeuron>2
%         for i = 1:length(seed_x)
%             if (seed_x(i)~=0) && (seed_y(i)~=0)
%                 seed_x(i) = seed_x(i)+randOffset(30);
%                 seed_y(i) = seed_y(i)+randOffset(30);
%             end
%         end
%     end