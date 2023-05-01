clear; close all;

% Problem parameters
% m<<n
m = 300;
n = 3000;

% Solution tolerance
sol_tol = 10^-5;


% Construct a feasible problem
x_feas = exprnd(1, n, 1); % x_feas >= 0  due to the problem constraint

A = randn(m, n);
b = A*x_feas;
c = exprnd(1, n, 1); % c >= 0  will guarantee that the generated instance of this problem will not be unbounded from bellow

% Solve the problem via CVX
% Find x_star and f_opt

cvx_begin quiet
	variable x(n)
	minimize( c.' * x )
	subject to
		A*x==b
		x>=0
cvx_end

x_star = cvx_optpnt.x;
f_opt = cvx_optval;


% Getting ready to run the ADMM algorithm.
% Do the proper setup to run the algorithm.

% The Augmented Lagrangian parameter rho
rho = 1;

% Run the ADMM algorithm
[x_est, f_k_admm, num_of_iters_admm] = admm_alg(c, A, b, rho, f_opt, sol_tol);

% Figure
figs(1) = figure;
semilogy(0:num_of_iters_admm, f_k_admm - f_opt);
hold on;
semilogy(0:num_of_iters_admm, f_k_admm - f_opt);
plot_setup({'ADMM', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{\\texttt{sol\\_tol} = %g}$', m, n, sol_tol)}, '$k$', '$f \left( \mathbf{x}^k \right) - f_{\mathrm{opt}}$');
