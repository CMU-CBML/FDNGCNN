function [phiK,tempr_new,conc_t_new,LAPpk] = phaseFieldModel(phi,tempr,conc_t,theta,cm,...
    epsilonb,delta,alphOverPix,gamma,theta_ori,C0,lenu,lenv,iter,r,g,tol,...
    lap,lap_flip,lap_id,M_phi,tau,dtime,phi_initial,tempr_initial,bcid,...
    kappa,Diff,alpha_t,beta_t,source_coeff,png_plot_invl)
   % calculating a and a*a' (aap) in the equation using theta and phi
    xtheta = cm.NuNv*theta;
    [a, ~, aap,~,~] = getEpsilonAndAap(epsilonb,delta,phi,xtheta,...
        cm.L_NuNv, cm.U_NuNv,cm.NuN1v,cm.N1uNv);
    
    NNtempr = cm.NuNv*tempr;
    NNct = cm.NuNv*conc_t;

    if(iter<=500)
        E = alphOverPix*atan(gamma*(1-NNtempr));
    else
        nnT = reshape(theta_ori,lenu*lenv,1);
        rMat = r*ones(lenu*lenv,1);
        gMat = g*ones(lenu*lenv,1);
        % adjust tip r g value
        rMat(nnT==1) = 5000;
        gMat(nnT==1) = 0;
        delta_L = rMat.*NNct - gMat;
        term_change = (regular_Heiviside_fun(delta_L));

        E = alphOverPix*atan(gamma*bsxfun(@times,term_change,1-NNtempr));

        if(mod(iter,png_plot_invl) == 0)
            phi_plot = reshape(cm.NuNv*phi,lenu,lenv);
            subplot(2,3,5);
            imagesc(reshape(E,lenu,lenv)+phi_plot);
            title(sprintf('E overlay with phi'));
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
        
        R = M_phi/tau*(out{1}-out{2}+out{3}+out{4});
        R = R*dtime-NNpk+(cm.NuNv*phi);
        dR = M_phi/tau*(t5+t6);
        dR = dR*dtime-cm.NuNv;

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

end