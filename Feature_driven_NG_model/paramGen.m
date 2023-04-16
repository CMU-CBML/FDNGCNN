close all;
clear all;
clc;

var_save_invl = 100;
png_save_invl = 100;
png_plot_invl = 100;
gc_sz = 2;
kappa = 2;
M_axon = 100;
M_neurites = 60;
delta = 0.2;

fid_sj = fopen('submit_jobs.sh','w');
fprintf(fid_sj,'#!/bin/sh\n\n');

for numNeuron = 1:7
    for numCase = 1:20
        caseName = "case"+numNeuron+"_"+numCase;
        status = mkdir(caseName);
        if status~=1
            error("Folder mkdir error! Possible dup.");
        else

            fprintf(fid_sj, 'cd ../'+string(caseName)+'\n');
            fprintf(fid_sj, 'sbatch main.job\n');

            cd(caseName);
            mkdir('data');
            mkdir("./data/initialization");
            fid = fopen('start_neuron_growth.m','w'); 
            fprintf(fid,'close all; clear all; clc;\n\n');
            fprintf(fid,'%% variable and png save frequency\n');
            formatSpec = 'var_save_invl = %d;\npng_save_invl = %d;\npng_plot_invl = %d;\n\n';
            fprintf(fid,formatSpec,var_save_invl,png_save_invl,png_plot_invl);
            fprintf(fid,'%% variable setup\n');
            formatSpec = 'numNeuron = %d;\ngc_sz = %d;\nkappa = %f;\nM_axon = %d;\nM_neurites = %d;\ndelta = %f;\n\n';
            fprintf(fid,formatSpec,numNeuron,gc_sz,kappa,M_axon,M_neurites,delta);
            fprintf(fid,'%% whether to continue simulation or start new\n');
            formatSpec = 'continue_simulation = %d;\n\n';
            fprintf(fid,formatSpec,0);    
            fprintf(fid,"addpath('../NeuronGrowth_IGA_collocation_algorithm');\n\n");
            fprintf(fid,'caseNum = '+string(numNeuron*numCase)+';\n\n');
            fprintf(fid,'main_randPos;\n');
            fclose(fid);

            n_cores = 4;
            fid = fopen('main.job','w'); 
            fprintf(fid, '#!/bin/sh\n\n');
            fprintf(fid, '#SBATCH --ntasks-per-node='+string(n_cores)+'\n');
            fprintf(fid, '#SBATCH --partition=RM-shared\n');
            fprintf(fid, '#SBATCH --mem-per-cpu=2000\n');
            fprintf(fid, '#SBATCH --job-name='+string(caseName.replace('case',''))+'h\n');
            fprintf(fid, '#SBATCH --output='+string(caseName)+'_log_%%j.out'+'\n');
            fprintf(fid, '#SBATCH --time=48:00:00\n');
            fprintf(fid, '#SBATCH --mail-type=ALL\n');
            fprintf(fid, '#SBATCH --mail-user=kuanrenq@andrew.cmu.edu\n\n');
            fprintf(fid, 'module load matlab\n');
            run_file = "'start_neuron_growth'";
            fprintf(fid, 'srun --exclusive -n 1 -c '+string(n_cores)+' matlab -nodisplay -nodesktop -nosplash -r "run('+run_file+'); exit;"\n\n');
            fprintf(fid, '#-- exit\n');
            fprintf(fid, '#\n');
            fclose(fid);

            fid = fopen('continue_neuron_growth.m','w'); 
            fprintf(fid,'close all; clear all; clc;\n\n');
            fprintf(fid,'%% whether to continue simulation or start new\n');
            formatSpec = 'continue_simulation = %d;\n\n';
            fprintf(fid,formatSpec,1);    
            fprintf(fid,"addpath('../NeuronGrowth_IGA_collocation_algorithm');\n\n");
            fprintf(fid,'main_randPos;\n');
            fclose(fid);

            n_cores = 4;
            fid = fopen('main_continue.job','w'); 
            fprintf(fid, '#!/bin/sh\n\n');
            fprintf(fid, '#SBATCH --ntasks-per-node='+string(n_cores)+'\n');
            fprintf(fid, '#SBATCH --partition=RM-shared\n');
            fprintf(fid, '#SBATCH --mem-per-cpu=2000\n');
            fprintf(fid, '#SBATCH --job-name='+string(caseName.replace('case',''))+'hc\n');
            fprintf(fid, '#SBATCH --output='+string(caseName)+'_log_%%j.out'+'\n');
            fprintf(fid, '#SBATCH --time=48:00:00\n');
            fprintf(fid, '#SBATCH --mail-type=ALL\n');
            fprintf(fid, '#SBATCH --mail-user=kuanrenq@andrew.cmu.edu\n\n');
            fprintf(fid, 'module load matlab\n');
            run_file = "'continue_neuron_growth'";
            fprintf(fid, 'srun --exclusive -n 1 -c '+string(n_cores)+' matlab -nodisplay -nodesktop -nosplash -r "run('+run_file+'); exit;"\n\n');
            fprintf(fid, '#-- exit\n');
            fprintf(fid, '#\n');
            fclose(fid);

            cd('../');
        end
    end
end
fclose(fid_sj);



