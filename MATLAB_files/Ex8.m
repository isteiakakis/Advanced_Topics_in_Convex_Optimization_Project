clear; close all;

% Problem parameters
m = 100;
n = 10;
lambda = 0.5;

% Termination condition
% The condition is  f_opt <= c*f(x_k)  iff  f_opt - f(x_k) <= (c-1)*f(x_k)
c = 1.01;


% Generate A, b
A = randn(m, n);
b = randn(m, 1);

% Solve the problem via CVX
% Find x_star and F_opt

cvx_begin quiet
	variable x(n)
	minimize( norm(A*x-b, 1) + lambda * norm(x, 1) )
cvx_end

x_star = cvx_optpnt.x;
F_opt = cvx_optval;

%% Question a

% Getting ready to run the subgradient descent algorithm using Polyak and dynamic stepsize.
% Do the proper setup to run the algorithm.

% Cost function and subgradients
f = @(x)  norm(A*x-b, 1);
f_subgrad = @(x)  l1_norm_aff_subgrad(A, -b, x);
g = @(x)  norm(x, 1);
g_subgrad = @(x)  l1_norm_aff_subgrad(1, 0, x);
F = @(x)  f(x) + lambda*g(x);
F_subgrad = @(x)  f_subgrad(x) + lambda * g_subgrad(x);

% Initial point for the algorithms
x_0 = randn(n, 1);

% Run subgradient descent with Polyak stepsize
[~, F_k_polyak, num_of_iters_polyak] = projected_subgradient_alg(F, F_subgrad, @(x) x, x_0, F_opt, NaN, (c-1)*F_opt, 'polyak');

% Run subgradient descent with dynamic stepsize
[~, F_k_dynamic, num_of_iters_dynamic] = projected_subgradient_alg(F, F_subgrad, @(x) x, x_0, F_opt, NaN, (c-1)*F_opt, 'dynamic');


%% Question b

% Getting ready to run the proximal subgradient algorithm.
% Do the proper setup to run the algorithm.

% Run the proximal subgradient algorithm
[~, F_k_prox, num_of_iters_prox] = proximal_subgradient_alg(A, b, lambda, x_0, F_opt, (c-1)*F_opt);

% Figure
figs(1) = figure;
semilogy(0:num_of_iters_polyak, F_k_polyak - F_opt);
hold on;
semilogy(0:num_of_iters_dynamic, F_k_dynamic - F_opt);
semilogy(0:num_of_iters_prox, F_k_prox - F_opt);
plot_setup({'Projected subgradient vs Proximal subgradient', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{\\lambda = %g}$, $\\mathbf{c = %g}$', m, n, lambda, c)}, '$k$', '$f \left( \mathbf{x}^k \right) - f_{\mathrm{opt}}$', 'Projected subgr. (Polyak)', 'Projected subgr. (dynamic)', 'Proximal subgr. (dynamic)');


%% Question c

% Getting ready to run S-FISTA (i.e. FISTA after smoothing the first part of the cost function).
% Do the proper setup to run the algorithm.

% Check out S-FISTA from section 10.8.4.

% Solution tolerance
sol_tol = (c-1)*F_opt;

% Compute mu, L_tilde
alpha = sigma_max(A)^2;
beta = m/2;
mu = sol_tol/(2*beta);
L_tilde = alpha/mu;

% Gradient of the smoothed part of the cost function
f_smoothed_grad = @(x)  sum_huber_scalar_affine_grad(mu, A, b, x);

% Run S-FISTA (which is FISTA on cost function with the first part smoothed; partially smoothed cost function)
[~, F_k_sfista_partially] = fista(F, f_smoothed_grad, lambda, L_tilde, x_0, F_opt, sol_tol);


%% Question d

% Getting ready to run S-FISTA after smoothing the whole cost function (i.e. FISTA on the wholy smoothed cost function with no proximal step).
% Do the proper setup to run the algorithm.

% Check out S-FISTA from section 10.8.4.

% Compute L_tilde
alpha = sigma_max(A)^2 + lambda;
beta = m/2 + lambda * n/2;
mu = sol_tol/(2*beta);
L_tilde = alpha/mu;

% Partially smoothed cost function
F_fully_smoothed = @(x)  sum_huber_scalar_affine(mu, A, b, x) + lambda * sum_huber_scalar_affine(mu, eye(n), zeros(n, 1), x);

% Gradient of the smoothed cost function
F_fully_smoothed_grad = @(x)  sum_huber_scalar_affine_grad(mu, A, b, x) + lambda * sum_huber_scalar_affine_grad(mu, eye(n), zeros(n, 1), x);

% Run S-FISTA after smoothing the whole cost function (which is FISTA on the smoothed cost function with no proximal step, that is lambda=0; fully smoothed cost function)
[~, F_k_sfista_fully] = fista(F, F_fully_smoothed_grad, 0, L_tilde, x_0, F_opt, sol_tol);

% Figure
figs(2) = figure;
semilogy(0:length(F_k_sfista_partially)-1, F_k_sfista_partially - F_opt);
hold on;
semilogy(0:length(F_k_sfista_fully)-1, F_k_sfista_fully - F_opt);
plot_setup({'S-FISTA: partially vs fully smoothed cost function', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{\\lambda = %g}$, $\\mathbf{c = %g}$', m, n, lambda, c)}, '$k$', '$f \left( \mathbf{x}^k \right) - f_{\mathrm{opt}}$', 'Partially smoothed', 'Fully smoothed');

% Figure (all together)
figs(3) = figure;
semilogy(0:num_of_iters_polyak, F_k_polyak - F_opt);
hold on;
semilogy(0:num_of_iters_dynamic, F_k_dynamic - F_opt);
semilogy(0:num_of_iters_prox, F_k_prox - F_opt);
semilogy(0:length(F_k_sfista_partially)-1, F_k_sfista_partially - F_opt);
semilogy(0:length(F_k_sfista_fully)-1, F_k_sfista_fully - F_opt);
plot_setup({'Projected subgr. vs Proximal subgr. vs S-FISTA', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{\\lambda = %g}$, $\\mathbf{c = %g}$', m, n, lambda, c)}, '$k$', '$f \left( \mathbf{x}^k \right) - f_{\mathrm{opt}}$', 'Projected subgr. (Polyak)', 'Projected subgr. (dynamic)', 'Proximal subgr. (dynamic)', 'S-FISTA: partially smoothed', 'S-FISTA: fully smoothed');
