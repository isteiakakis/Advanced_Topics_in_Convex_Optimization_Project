clear; close all;

% For figures
linewidth = .5;

% Problem parameters
% m > n
m = 300;
n = 30;

% Termination condition used in questions b, c
% The condition is  f_opt <= c*f(x_k)  iff  f_opt - f(x_k) <= (c-1)*f(x_k)
c = 1.01;


% Generate A, b
% Uncomment the commands of either Case I or Case II

% Case I: Gaussian distribution (expected value 0, standard deviation 1)
A = randn(m, n);
b = randn(m, 1);

% Case II: Uniform distribution (expected value 0, standard deviation 1)
%A = sqrt(12) * (rand(m, n) - 0.5);
%b = sqrt(12) * (rand(m, 1) - 0.5);


%% Question a

% Solve the problem via CVX
% Find x_star and f_opt

fprintf('Question a\n');

cvx_begin quiet
	variable x(n)
	minimize( norm(A*x-b, Inf) )
cvx_end

x_star = cvx_optpnt.x;
f_opt = cvx_optval;

fprintf('Done\n\n');


%% Question b

% Getting ready to run the subgradient descent algorithm using Polyak stepsize.
% Do the proper setup to run the algorithm.

fprintf('Question b\n');

% The cost function
f = @(x)  norm(A*x - b, Inf);

% The subgradient of the cost function (weak result)
f_subgrad = @(x)  inf_norm_aff_subgrad(A, -b, x);

% Initial point for the algorithms
x_0 = randn(n, 1);

% Run the projected subgradient algorithm without projection using the Polyak stepsize
[~, f_k_polyak, polyak_num_of_iters] = projected_subgradient_alg(f, f_subgrad, @(x) x, x_0, f_opt, NaN, (c-1)*f_opt, 'polyak');

% Get the best achieved values
f_k_best_polyak = best_achieved_values(f_k_polyak);

% Figure
figs(1) = figure;
semilogy(0:polyak_num_of_iters, f_k_best_polyak - f_opt, 'Linewidth', linewidth);
plot_setup({'Subgradient descent with Polyak stepsize,', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{c = %g}$', m, n, c)}, ...
	'$k$', '$f_{\mathrm{best}}^k - f_{\mathrm{opt}}$');

fprintf('Done\n\n');


%% Question c

% Getting ready to run the subgradient descent algorithm using Polyak stepsize.
% Do the proper setup to run the algorithm.

fprintf('Question c\n');

% The cost function and its subgradient are the same as in question b

% The initial point is also the same as in question b

% Run the subgradient descent algorithm without projection using the dynamic stepsize
[~, f_k_dynamic, dynamic_num_of_iters] = projected_subgradient_alg(f, f_subgrad, @(x) x, x_0, f_opt, NaN, (c-1)*f_opt, 'dynamic');

% Get the best achieved values
f_k_best_dynamic = best_achieved_values(f_k_dynamic);

% Figure
figs(2) = figure;
semilogy(0:dynamic_num_of_iters, f_k_best_dynamic - f_opt, 'Linewidth', linewidth);
plot_setup({'Subgradient descent with dynamic stepsize,', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{c = %g}$', m, n, c)}, ...
	'$k$', '$f_{\mathrm{best}}^k - f_{\mathrm{opt}}$');

fprintf('Done\n\n');


%% Question d

% Compare the results in the same figure

fprintf('Question d\n');

% Figure
figs(3) = figure;
semilogy(0:polyak_num_of_iters, f_k_best_polyak - f_opt, 'Linewidth', linewidth);
hold on;
semilogy(0:dynamic_num_of_iters, f_k_best_dynamic - f_opt, 'Linewidth', linewidth);
plot_setup({'Subgradient descent, Polyak and dynamic stepsize,', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{c = %g}$', m, n, c)}, ...
	'$k$', '$f_{\mathrm{best}}^k - f_{\mathrm{opt}}$', 'Polyak', 'Dynamic');

fprintf('Done\n\n');


%% Question e

% Compute and plot the upper bound of the convergence diagram of the above algorithms

fprintf('Question e\n');

% Compute L_f as shown in report
L_f = sigma_max(A) * sqrt(m);

% Figure
figs(4) = figure;
polyak_plot = semilogy(0:polyak_num_of_iters, f_k_best_polyak - f_opt, 'Linewidth', linewidth);
hold on;
semilogy(0:dynamic_num_of_iters, f_k_best_dynamic - f_opt, 'Linewidth', linewidth);
semilogy(0:polyak_num_of_iters, (L_f * norm(x_0-x_star)) ./ sqrt((0:polyak_num_of_iters)+1), '--', 'Linewidth', linewidth, 'Color', polyak_plot.Color);
plot_setup({'Subgradient descent, Polyak and dynamic stepsize,', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{c = %g}$', m, n, c)}, '$k$', '$f_{\mathrm{best}}^k - f_{\mathrm{opt}}$', 'Polyak', 'Dynamic', 'Polyak''s upper bound');

fprintf('Done\n\n');


