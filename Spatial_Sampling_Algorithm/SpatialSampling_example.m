%--------------------------------------------------------------------------
% Spatial Sampling example code
%
% Written by Giovanni Braglia, 2024
% University of Modena and Reggio Emilia
% 
% tested on MATLAB R2022a
%--------------------------------------------------------------------------

clear; clc; close all;

% Load data from Panda Co-Manipulation Dataset
%---------------------------------------------
load("symbol_data.mat");
Ts = 0.001; % Sampling time of recording
i = 3; % i = [0,1,...,5]
L = length( symbol_data(i).pos );
T = Ts*L; % Duration of the recording
t = linspace( 0, T, L );
pos = symbol_data(i).pos; % Position data

% Spatial Sampling
%---------------------------------------------
delta = 0.005; % Spatial Sampling's interval
gamma = delta*1e-3; % Delta tolerance
out = SpatialSampling( t, pos, delta, gamma );

tn = out.tn;
sn = out.sn;
xn = out.xn;

figure;

subplot(3,1,[1,2]);
%------------------ 
grid on; hold on;
plot( symbol_data(i).pos(1,:), symbol_data(i).pos(2,:), 'k-', LineWidth=2, DisplayName='Original' );
plot( xn(:,1), xn(:,2), 'r--', LineWidth=2, DisplayName='Filtered' );

legend('Location','best'); 
xlabel( '$x$[m]','FontSize',14,'Interpreter','latex');
ylabel( '$y$[m]','FontSize',14,'Interpreter','latex');

subplot(3,1,3);
%------------------
plot( tn, sn, 'r-', LineWidth=2 );
grid on;
xlabel( '$t$[s]','FontSize',14,'Interpreter','latex');
ylabel( '$s$[m]','FontSize',14,'Interpreter','latex');


