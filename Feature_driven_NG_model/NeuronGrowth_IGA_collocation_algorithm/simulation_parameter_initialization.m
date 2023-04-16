%% Simulation Parameter Initialization
% time stepping variables
dtime = 1e-2;
iter = 0;
end_iter = 100000; % large enough number

% % setup domain size outside main for modularity (main.m can remain the same
% for different simulation cases)
neuron_domain_setup

% B-spline curve order (U,V direction)
p = 3;
q = 3;

knotvectorU = [0,0,0,linspace(0,Nx,Nx+1),Nx,Nx,Nx].';
knotvectorV = [0,0,0,linspace(0,Ny,Ny+1),Ny,Ny,Ny].';

% setting lenu lenv this way for easier access to ghost nodes later on
lenu = length(knotvectorU)-2*(p-1);
lenv = length(knotvectorV)-2*(p-1);

% neuron growth variables
aniso = 6;
if exist('kappa','var') == 0;kappa = 2;end % kappa= 2;
alph = 0.9; % changing name to alph cause alpha is a function
pix=4.0*atan(1.0);
alphOverPix = alph/pix;
gamma = 10.0;
tau = 0.3;
if exist('M_phi','var') == 0;M_phi = 60;end
M_theta = 0.5*M_phi;
s_coeff = 0.007;
if exist('delta','var') == 0;delta = 0.1;end
if exist('epsilonb','var') == 0;epsilonb = 0.04;end

% Tubulin parameters
if exist('r','var') == 0;r = 5;end
if exist('g','var') == 0;g = 0.1;end
if exist('alpha_t','var') == 0;alpha_t = 0.001;end
if exist('beta_t','var') == 0;beta_t = 0.001;end
if exist('Diff','var') == 0;Diff = 4;end
if exist('source_coeff','var') == 0;source_coeff = 15;end

% tolerance for NR iterations in phi equation
tol = 1e-4;
tip_state = 0;

% Seed size
% seed_radius = 10;
seed_radius = 20;
% initializing phi and concentration based on neuron seed position
[phi,conct,seed_x,seed_y] = initialize_neurite_growth_rand2(seed_radius,...
    lenu,lenv,numNeuron);
save('./data/initialization/seed_x','seed_x');
save('./data/initialization/seed_y','seed_y');

% Expanding domain parameters
% BC_clearance = 25;
BC_clearance = 25;
expd_sz = 10;

if exist('gc_sz','var') == 0;gc_sz = 4;end % gc_sz = 4;

%% Iterating Variable Initialization
% constructing collocation basis
order_deriv = 2;    % highest order of derivatives to calculate
[cm,size_collpts] = collocationDers(knotvectorU,p,knotvectorV,q,...
    order_deriv);
lap = cm.N2uNv + cm.NuN2v;
[lap_flip, lap_id] = extract_diags(lap);

phi = cm.NuNv\phi;
conc_t = cm.NuNv\conct;

% initializing theta and temperature
theta=cm.NuNv\reshape(rand(lenu,lenv),lenu*lenv,1);
theta_ori = zeros(lenu,lenv);
tempr = zeros([lenu*lenv,1]);

% theta does not evolve over time, only need to compute initially or 
% expanding domain magnitude of theta gradient
mag_grad_theta = sqrt((cm.N1uNv*theta).*(cm.N1uNv*theta)+...
    (cm.NuN1v*theta).*(cm.NuN1v*theta));
C0 = 0.5+6*s_coeff*mag_grad_theta;
    
% initializing initial phi,theta,tempr for boundary condition (Dirichlet)
phi_initial = reshape(phi,lenu,lenv);
theta_initial = reshape(theta,lenu,lenv);
tempr_initial = reshape(tempr,lenu,lenv);
for i = 2:lenu-1
    for j = 2:lenv-1
        phi_initial(i,j) = 0;
        theta_initial(i,j) = 0;
        tempr_initial(i,j) = 0;
    end
end
phi_initial = reshape(phi_initial,lenu*lenv,1);
theta_initial  = reshape(theta_initial,lenu*lenv,1);
tempr_initial  = reshape(tempr_initial,lenu*lenv,1);
save('./data/initialization/phi_on_cp_initial','phi');
save('./data/initialization/theta_on_cp_initial','theta');
save('./data/initialization/tempr_on_cp_initial','tempr');

% plotting initial phi
set(gcf,'position',[100,100,800,400]);
colormap parula;

% binary ID for boundary location (define 4 edges)
% id = 1 means there is bc
bcid = zeros([lenu,lenv]);
for i = 1:lenu
    bcid(1,i) = 1;
    bcid(lenu,i) = 1;
    bcid(i,1) = 1;
    bcid(i,lenv) = 1;
end
bcid = reshape(bcid,lenu*lenv,1);

dist = zeros(lenu,lenv);

disp('Iterating Variable Initialization - Done!');
disp('******************************************************************');