%% IGA-collocation Implementation for 2D neuron growth
% Last edit date: 12/21/2022
% Kuanren Qian

%% Start Simulation Model
disp('******************************************************************');
disp('2D Phase-field Neuron Growth solver using IGA-Collocation');
disp('******************************************************************');

%% Initialization
% Simulation Parameter
if continue_simulation == 0
    rngSeed = datenum(datetime)/1e4+second(datetime)+caseNum
    save('./data/initialization/rngSeed.mat','rngSeed');
    rng(rngSeed);

    simulation_parameter_initialization

    % Stats from Ashlee's paper
    ashlee_stats_initialization
    perNeuronNeuriteLength = 0;
    theta_ori_phi = ones(lenu,lenv);
    prevDiv = 0;
    divLog = [0,0];

    M_phi = ones(lenu*lenv,1).*M_axon;

    xtheta = cm.NuNv*theta; %NNtheta
    N1Ntheta = cm.N1uNv*theta;
    NN1theta = cm.NuN1v*theta;
else
    load('./data/workspace.mat');
end


%% Transient iteration computation
disp('Starting Neuron Growth Model transient iterations...');
tic;
changePoints = [0,0];
kernel_circle = ones(20);
for i =1:20
    for j=1:20
        if sqrt((i-10)^2+(j-10)^2)>=9
            kernel_circle(i,j) = 0;
        end
    end
end

while iter < end_iter
    iter = iter + 1;
    if(mod(iter,100) == 0)
        fprintf('Progress: %.2d/%.2d\n',iter,end_iter);
        fprintf('Elapsed time: %.2f | Current time: %s\n',toc,...
            datestr(now,'mm/dd/yy-HH:MM:SS'));
    end

    % calculating a and a*a' (aap) in the equation using theta and phi
    [a, ~, aap,~,~] = getEpsilonAndAap(epsilonb,delta,phi,xtheta,...
        cm.L_NuNv, cm.U_NuNv,cm.NuN1v,cm.N1uNv);
    
    NNtempr = cm.NuNv*tempr;
    NNct = cm.NuNv*conc_t;

    if(iter<=100)
        E = alphOverPix*atan(gamma*(1-NNtempr));
    else
        nnT = reshape(theta_ori,lenu*lenv,1);
        rMat = r*ones(lenu*lenv,1);
        gMat = g*ones(lenu*lenv,1);
        % adjust tip r g value
        rMat(nnT==1) = 50;
        gMat(nnT==1) = 0;
        delta_L = rMat.*NNct - gMat;
        term_change = (regular_Heiviside_fun(delta_L));

        E = alphOverPix*atan(gamma*bsxfun(@times,term_change,1-NNtempr));

        if(mod(iter,png_plot_invl) == 0)
            phi_plot = reshape(cm.NuNv*phi,lenu,lenv);
            subplot(2,3,5);
            imagesc(reshape(E,lenu,lenv)+phi_plot);
            title(sprintf('E overlay phi %.2d',iter));
            axis square;
            colorbar;
            drawnow;
        end
    end

    %% Phi (Implicit Nonlinear NR method)
    % NR method initial guess (guess current phi)
    phiK = phi;
    % initial residual for NR method
    R = 2*tol;

    % splitted C0 from C1 because E mag_grad_theta dimension mismatch
    % during domain expansion. Compute here to fix conflict
    C1 = E-C0;

    % NR method calculation
    ind_check = 0;
    NNa = cm.NuNv*a;
    N1Na = cm.N1uNv*a;
    NN1a = cm.NuN1v*a;
    NNaap = cm.NuNv*aap;
    N1Naap = cm.N1uNv*aap;
    NN1aap = cm.NuN1v*aap;

    out = cell(4,1);
    dt_t = 0;
    while max(abs(R)) >= tol
        NNpk = cm.NuNv*phiK;
        N1Npk = cm.N1uNv*phiK;
        NN1pk = cm.NuN1v*phiK;
        N1N1pk = cm.N1uN1v*phiK;
        LAPpk = lap*phiK;
        
        % term a2
        out{1}  = 2*bsxfun(@times,bsxfun(@times,NNa,N1Na),N1Npk)+...
            bsxfun(@times,bsxfun(@times,NNa,NNa),LAPpk) ...
            +2*bsxfun(@times,bsxfun(@times,NNa,NN1a),NN1pk);
        % termadx
        out{2} = bsxfun(@times,N1Naap,NN1pk)+bsxfun(@times,NNaap,N1N1pk);
        %termady
        out{3} = bsxfun(@times,NN1aap,N1Npk)+bsxfun(@times,NNaap,N1N1pk);
        % termNL
        out{4} = -bsxfun(@times,bsxfun(@times,NNpk,NNpk),NNpk)+...
            bsxfun(@times,bsxfun(@times,(1-C1),NNpk),NNpk)+...
            bsxfun(@times,C1,NNpk);
        if dt_t==0 % these terms only needs to be calculated once
            % terma2_deriv
            t5 =  banded_dot_star((2*bsxfun(@times,NNa,N1Na)+N1Naap),...
                cm.N1uNv_flip,cm.N1uNv_id)+...
                banded_dot_star(bsxfun(@times,NNa,NNa),lap_flip,lap_id)+...
                banded_dot_star((2*bsxfun(@times,NNa,NN1a)-N1Naap),...
                cm.NuN1v_flip,cm.NuN1v_id);
        end
        % termNL_deriv
        temp = (-3*bsxfun(@times,NNpk,NNpk)+2*bsxfun(@times,(1-C1),NNpk)...
            +C1);
        t6 = banded_dot_star(temp, cm.NuNv_flip, cm.NuNv_id);
        
        R = M_phi./tau.*(out{1}-out{2}+out{3}+out{4});
        R = R*dtime-NNpk+(cm.NuNv*phi);
        dR = M_phi./tau.*(t5+t6);
        dR = dR.*dtime-cm.NuNv;

        % check residual and update guess
        R = R - dR*phi_initial;
        [dR, R] = StiffMatSetupBCID(dR, R,bcid,phi_initial);
        dp = dR\(-R);
        phiK = phiK + dp;
        
        max_phi_R = full(max(abs(R)));
        if (ind_check >= 100 || max(abs(R))>1e20)
            error('Phi NR method NOT converging!-Max residual: %.2d\n',...
                max_phi_R);
        end
        ind_check = ind_check + 1;
        dt_t = dt_t+dtime;
    end

    %% Temperature (Implicit method)
    temprLHS = cm.NuNv-3*dt_t*lap;
    temprRHS = kappa*(cm.NuNv*phiK-cm.NuNv*phi)+NNtempr;
    [temprLHS, temprRHS] = StiffMatSetupBCID(temprLHS, temprRHS,bcid,...
        tempr_initial);
    tempr_new = temprLHS\temprRHS;

    %% Tubulin concentration (Implicit method)
    NNp = cm.NuNv*phi;
    nnpk = round(NNpk);

    LAPpk = lap*phi;
    sum_lap_phi = sum(bsxfun(@times,LAPpk,LAPpk));

    term_diff = Diff*(banded_dot_star(N1Npk,cm.N1uNv_flip,cm.N1uNv_id)+...
        banded_dot_star(NNpk,lap_flip,lap_id)+...
        banded_dot_star(NN1pk,cm.NuN1v_flip,cm.NuN1v_id));
    term_alph = alpha_t*(banded_dot_star(N1Npk,cm.NuNv_flip,cm.NuNv_id)+...
        banded_dot_star(NNpk,cm.N1uNv_flip,cm.N1uNv_id)+...
        banded_dot_star(NN1pk,cm.NuNv_flip,cm.NuNv_id)+...
        banded_dot_star(NNpk,cm.NuN1v_flip,cm.NuN1v_id));
    term_beta = beta_t*banded_dot_star(NNpk,cm.NuNv_flip,cm.NuNv_id);
    term_source = source_coeff/sum_lap_phi*bsxfun(@times,LAPpk,LAPpk);

    conc_t_RHS = dtime/2*term_source-bsxfun(@times,NNct,(NNpk-NNp))+...
        bsxfun(@times,NNp,NNct);
    conc_t_LHS = bsxfun(@times,NNp,cm.NuNv)-dtime/2*(term_diff-term_alph...
        -term_beta);
    bcid_t = (~nnpk);
    [conc_t_LHS, conc_t_RHS] = StiffMatSetupBCID(conc_t_LHS, conc_t_RHS,...
        bcid_t,zeros(lenu*lenv,1));
    conc_t_new = conc_t_LHS\conc_t_RHS;

    %% iteration update
    % update variables in this iteration
    phi = phiK;
    tempr = tempr_new;
    conc_t = conc_t_new;

  %% Check for domain expansion
    if mod(iter,100) == 0
        edgeID = zeros(1,4);
        if max(max(phi_plot(1:BC_clearance,:))) > 0.5
            edgeID(1) = 1; % Top
        end
        if max(max(phi_plot(:,1:BC_clearance))) > 0.5
            edgeID(2) = 1; % Left
        end
        if max(max(phi_plot(end-BC_clearance:end,:))) > 0.5
            edgeID(3) = 1; % Bottom
        end
        if max(max(phi_plot(:,end-BC_clearance:end))) > 0.5
            edgeID(4) = 1; % Right
        end

        if any(edgeID)
            disp('------------------------------------------------------');
            disp('Expanding Domain...');
            disp(edgeID);
            [phi,phi_mask,theta_ori_phi,M_phi,theta,conc_t,tempr,LAPpk,phi_initial,...
                theta_initial,tempr_initial,bcid,seed_x,seed_y,changePoints] = ...
                expandDomain(phiK,phi_mask,theta_ori_phi,M_phi,theta,conc_t_new,tempr_new,...
                LAPpk,cm.NuNv,edgeID,expd_sz,seed_x,seed_y,changePoints);
            lenu = sqrt(length(phi)); lenv = lenu;
            M_phi = reshape(M_phi,lenu*lenv,1);

            Nx = lenu-3; Ny = lenv-3;
            knotvectorU = [0,0,0,linspace(0,Nx,Nx+1),Nx,Nx,Nx].';
            knotvectorV = [0,0,0,linspace(0,Ny,Ny+1),Ny,Ny,Ny].';

            [cm, size_collpts] = collocationDers(knotvectorU,p,...
                knotvectorV,q,order_deriv);
            lap = cm.N2uNv + cm.NuN2v;
            [lap_flip, lap_id] = extract_diags(lap);
            theta = sparse(cm.NuNv\reshape(theta,lenu^2,1));
            xtheta = cm.NuNv*theta; %NNtheta
            N1Ntheta = cm.N1uNv*theta;
            NN1theta = cm.NuN1v*theta;
            mag_grad_theta = sqrt(N1Ntheta.*N1Ntheta+...
                NN1theta.*NN1theta);
            C0 = 0.5+6*s_coeff*mag_grad_theta;
            [Y,X] = meshgrid(1:lenu,1:lenv); %[Y,X] matches other variables

            toc
            disp('------------------------------------------------------');
        end
    end

    %% Growth stage (div) operations
    growth_stage_operations

    %% Plotting
    if(mod(iter,png_plot_invl)==0 || iter == 0)
        phi_plot = reshape(cm.NuNv*phi,lenu,lenv);
        tempr_plot = reshape(cm.NuNv*tempr,lenu,lenv);
        theta_plot = reshape(cm.NuNv*theta,lenu,lenv);
        conct_plot = reshape(cm.NuNv*conc_t,lenu,lenv);

        subplot(2,3,1);
        imagesc(phi_plot); hold on;
        scatter(lenv/2+seed_y,lenu/2+seed_x,'rx'); hold off;
        title(sprintf('Phi at iteration = %.2d',iter));
        axis square;
        colorbar;

%         subplot(2,3,2);
%         imagesc(tempr_plot);
%         title(sprintf('T at iteration = %.2d',iter));
%         axis square;
%         colorbar;

        subplot(2,3,2);
        imagesc(tip);
        title(sprintf('T at iteration = %.2d',iter));
        axis square;
        colorbar;

    
        % plot current iteration
        drawnow;

        % save picture
        if(mod(iter,png_save_invl) == 0)
            try
                saveas(gcf,sprintf('./data/NeuronGrowth_%.2d.png',iter));
            catch
                fprintf('png write error skipped.\n');
            end
        end

        if(mod(iter,var_save_invl)==0 || iter == 0)
            save(sprintf('./data/phi_%2d',iter),'phi_plot');
            save(sprintf('./data/tempr_%2d',iter),'tempr_plot');
            save(sprintf('./data/theta_%2d',iter),'theta_plot');
            save(sprintf('./data/conct_%2d',iter),'conct_plot');
            save(sprintf('./data/tips_%2d',iter),'theta_ori');
        end
    
        if(mod(iter,2500)==0)
            save(sprintf('./data/workspace_%2d',iter),'-v7.3');
        end
        if(mod(iter,500)==0)
            save(sprintf('./data/workspace',iter),'-v7.3');
        end
    end


end

disp('******************************************************************');
disp('All simulations complete!\n');
disp('******************************************************************');