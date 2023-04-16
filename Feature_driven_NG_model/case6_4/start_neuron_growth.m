close all; clear all; clc;

% variable and png save frequency
var_save_invl = 100;
png_save_invl = 100;
png_plot_invl = 100;

% variable setup
numNeuron = 6;
gc_sz = 2;
kappa = 2.000000;
M_axon = 100;
M_neurites = 60;
delta = 0.200000;

% whether to continue simulation or start new
continue_simulation = 0;

addpath('../NeuronGrowth_IGA_collocation_algorithm');

caseNum = 24;

main_randPos;
