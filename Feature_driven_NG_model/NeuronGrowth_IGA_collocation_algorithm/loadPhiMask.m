function [phi_mask] = loadPhiMask(var_save_invl)
    load_state = 0;
    load_idx = 0;
%     iter = 100;
    iter = 800;
    while load_state == 0
        try
%             phi_mask = smoothNeurite(load(sprintf('./data/phi_%2d.mat',iter)).phi_plot,sigma);
            phi_mask = load(sprintf('./data/phi_%2d.mat',iter)).phi_plot;
            load_state = 1;
        catch
            iter = iter-var_save_invl;
            load_idx = load_idx+1;
        end
        if load_idx > 20
            fprintf('Failed to load.\n');
            break;
        end
    end
end