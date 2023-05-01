clear; close all;

% Problem parameters
% m <= n because A must be full row rank so as the constraint A*x=b has feasible points
m = 198;
n = 200;

% Termination condition (solution tolerance)
% The condition is  f_opt - f(x_k) <= sol_tol
sol_tol = 10^-4;


% Generate A, b
A = randn(m, n);
b = randn(m, 1);


%% Question a

% Solve the problem via CVX
% Find x_star and h_opt

cvx_begin quiet
	variable x(n)
	minimize( norm(x, 1) )
	subject to
		A*x == b
cvx_end

x_star = cvx_optpnt.x;
h_opt = cvx_optval;


%% Question b

% Getting ready to run S-FISTA (i.e. FISTA after smoothing the cost function).
% Do the proper setup to run the algorithm.

% Check out S-FISTA from section 10.8.4.

% Initial point for the algorithms
x_0 = randn(n, 1);

% Cost function
h = @(x)  norm(x, 1);

% Compute mu, L_tilde
alpha = 1;
beta = n/2;
mu = sol_tol/(2*beta);
L_tilde = alpha/mu;

% Gradient of the smoothed cost function
h_smoothed_grad = @(x)  sum_huber_scalar_affine_grad(mu, eye(n), zeros(n, 1), x);

% Run S-FISTA after smoothing the cost function (which is FISTA on the smoothed cost function with no proximal step, that is lambda=0)
[~, h_k_sfista] = fista(h, h_smoothed_grad, 0, L_tilde, x_0, h_opt, sol_tol, NaN, @(x)  project_onto_affine_set(A, b, x));

% Number of S-FISTA iterations.
sfista_num_of_iters = length(h_k_sfista)-1;


%% Question c

% Getting ready to run the subgradient descent algorithm using dynamic stepsize.
% Do the proper setup to run the algorithm.

% The cost function is the same as above

% The subgradient of the cost function
h_subgrad = @(x)  l1_norm_aff_subgrad(1, 0, x);

% Run the projected subgradient algorithm using the dynamic stepsize
[~, h_k_subgrad] = projected_subgradient_alg(h, h_subgrad, @(x)  project_onto_affine_set(A, b, x), x_0, NaN, sfista_num_of_iters, NaN, 'dynamic');


%% Question d

% Figure
figs(1) = figure;
semilogy(0:sfista_num_of_iters, h_k_sfista - h_opt);
hold on;
semilogy(0:sfista_num_of_iters, h_k_subgrad - h_opt);
plot_setup({'S-FISTA vs Projected subgradient', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{\\texttt{sol\\_tol} = %g}$', m, n, sol_tol)}, '$k$', '$f \left( \mathbf{x}^k \right) - f_{\mathrm{opt}}$', 'S-FISTA', 'Projected subgradient');
