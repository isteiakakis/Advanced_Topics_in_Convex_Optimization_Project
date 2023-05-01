clear; close all;

% For figures

% Problem parameters
m = 20;
n = 200;

% Termination condition for the algorithms
% The condition is  f_opt <= c*f(x_k)  iff  f_opt - f(x_k) <= (c-1)*f(x_k)
c = 1.0001;

%% Question e
% Generate A, b

% Random A
A = randn(m, n);

% Uncomment the commands of either Case I or Case II

% Case I: random b
%b = randn(m, 1);

% Case II: b = A*x_true + e
noise_std = 1; % Noise standard deviation
b = A * randn(n, 1) + noise_std * randn(m, 1);


%% Question a

% Solve the problem via CVX
% Find x_star and f_opt

cvx_begin quiet
	variable x(n)
	minimize( norm(A*x-b, 1) )
	subject to
		x >= 0
		sum(x) == 1
cvx_end

x_star = cvx_optpnt.x;
f_opt = cvx_optval;


%% Question b

% Getting ready to run the subgradient descent algorithm using Polyak and dynamic stepsize.
% Do the proper setup to run the algorithm.

% Define the cost function and the subgradient function
f = @(x)  norm(A*x - b, 1);
f_subgrad = @(x)  l1_norm_aff_subgrad(A, -b, x);

% Initial point for the algorithms
x_0 = 1/n * ones(n, 1);

% Run the projected subgradient algorithm using the dynamic stepsize (check out Example 9.19)
[~, f_k_proj_subgrad, num_of_iters_proj_subgrad] = projected_subgradient_alg(f, f_subgrad, @project_onto_unit_simplex, x_0, f_opt, NaN, (c-1)*f_opt, 'dynamic', sqrt(2));

% Get the best achieved values
f_k_best_proj_subgrad = best_achieved_values(f_k_proj_subgrad);


%% Question c

% Run the mirror descent algorithm
[~, f_k_mirror, num_of_iters_mirror] = mirror_descent_alg(A, b, x_0, f_opt, (c-1)*f_opt);

% Get the best achieved values
f_k_best_mirror = best_achieved_values(f_k_mirror);


%% Question d

% Figure
figs(1) = figure;
semilogy(0:num_of_iters_proj_subgrad, f_k_best_proj_subgrad - f_opt);
hold on;
semilogy(0:num_of_iters_mirror, f_k_best_mirror - f_opt);
plot_setup({'Projected subgradient vs Mirror descent', sprintf('with dynamic stepsize, $\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{c = %g}$', m, n, c)}, ...
	'$k$', '$f_{\mathrm{best}}^k - f_{\mathrm{opt}}$', 'Projected subgradient', 'Mirror descent');

