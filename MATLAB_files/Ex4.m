clear; close all;

% For figures
linewidth = .5;

% Problem parameters
% m > n
m = 300;
n = 30;

% Termination condition used in questions b, c; the number of iterations for questions f, g is determined by the number of iterations done in question c
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
	minimize( norm(A*x-b, 1) )
cvx_end

x_star = cvx_optpnt.x;
f_opt = cvx_optval;

fprintf('Done\n\n');


%% Question b

% Getting ready to run the subgradient descent algorithm using Polyak stepsize.
% Do the proper setup to run the algorithm.

fprintf('Question b\n');

% The cost function
f = @(x)  norm(A*x - b, 1);

% The subgradient of the cost function (weak result)
f_subgrad = @(x)  l1_norm_aff_subgrad(A, -b, x);

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


%% Question f

% Getting ready to run the stochastic subgradient descent algorithm using Polyak stepsize.
% Do the proper setup to run the algorithm.

fprintf('Question f\n');

% The cost function is the same as in questions b, c

% Subgradient of f_i - weak result
f_subgrad_i = @(i, x)  l1_norm_aff_subgrad(A(i, :), -b(i), x);

% Arbitrary choice for theta
theta = 1/2 * norm(x_0 - x_star)^2;

% Compute L_f_i in the same way as L_f
L_f_i = zeros(m, 1);
for i=1:m
	L_f_i(i) = sigma_max(A(i, :));
end

% Run the stochastic projected subgradient algorithm without projection for  dynamic_num_of_iters  number of iterations
[~, f_k_stochastic] = stochastic_projected_subgradient_alg(f, f_subgrad_i, @(x) x, L_f_i, m, theta, x_0, dynamic_num_of_iters); % x_star  is used to compute theta for the algorithm

% Figure
figs(5) = figure;
semilogy(0:dynamic_num_of_iters, f_k_stochastic - f_opt, 'Linewidth', linewidth);
plot_setup({'Stochastic subgradient,', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{c = %g}$', m, n, c)}, '$k$', '$f \left( \mathbf{x}^{km} \right) - f_{\mathrm{opt}}$');

fprintf('Done\n\n');


%% Question g

% Getting ready to run the incremental subgradient descent algorithm using Polyak stepsize.
% Do the proper setup to run the algorithm.

fprintf('Question g\n');

% The cost function is the same as in questions b, c, f

% The subgradient of f_i(x) is the same as in question f

% Run the incremental projected subgradient algorithm without projection for  dynamic_num_of_iters  number of iterations
[~, f_k_incremental] = incremental_projected_subgradient_alg(f, f_subgrad, f_subgrad_i, @(x) x, m, x_0, dynamic_num_of_iters);

% Figure
figs(6) = figure;
semilogy(0:dynamic_num_of_iters, f_k_incremental - f_opt, 'Linewidth', linewidth);
plot_setup({'Incremental subgradient with dynamic stepsize,', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{c = %g}$', m, n, c)}, '$k$', '$f \left( \mathbf{x}^{km} \right) - f_{\mathrm{opt}}$');

fprintf('Done\n\n');


%% Question h
fprintf('Question h\n');

% Put all plots together

% Figure
figs(7) = figure;
polyak_plot = semilogy(0:polyak_num_of_iters, f_k_best_polyak - f_opt, 'Linewidth', linewidth);
hold on;
dynamic_plot = semilogy(0:dynamic_num_of_iters, f_k_best_dynamic - f_opt, 'Linewidth', linewidth);
stochastic_plot = semilogy(0:dynamic_num_of_iters, f_k_stochastic - f_opt, 'Linewidth', linewidth);
incremental_plot = semilogy(0:dynamic_num_of_iters, f_k_incremental - f_opt, 'Linewidth', linewidth);

plot_setup({'All plots together,', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{c = %g}$', m, n, c)}, '$k$', '', ...
	'Subgradient descent, Polyak, $f_{\mathrm{best}}^k - f_{\mathrm{opt}}$', ...
	'Subgradient descent, dynamic, $f_{\mathrm{best}}^k - f_{\mathrm{opt}}$', ...
	'Stochastic subgr., dynamic, $f \left(\mathbf{x}^{km}\right) - f_{\mathrm{opt}}$', ...
	'Incremental subgr., dynamic, $f \left(\mathbf{x}^{k}\right) - f_{\mathrm{opt}}$');

% Appear the plots in reverse order (for appearance purposes)
uistack(stochastic_plot, 'top');
uistack(dynamic_plot, 'top');
uistack(polyak_plot, 'top');

fprintf('Done\n\n');

