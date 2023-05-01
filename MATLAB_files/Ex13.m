clear; close all;

markersize = 4;

% Problem parameters
n = 200;
lambda = 1;
noise_std = 0.1; % Noise standard deviation


% Uncomment the commands of either Case I or Case II

% Case I: Create a piecewise constant vector
d = zeros(n, 1);
d(1:floor(n/4)) = 1;
d(floor(n/4)+1:floor(n/2)) = 3;
d(floor(n/2)+1:floor(3*n/4)) = 0;
d(floor(3*n/4)+1:end) = 2;

% Case II: Create a smooth vector
%d = ((-n/2:n/2-1)*0.01).'.^2;


% Add noise to the initial vector
d_noised = d + noise_std*randn(n, 1);

% The D matrix (check out the Section 12.4.3)
D = [eye(n-1) , zeros(n-1, 1)] + [zeros(n-1, 1) , -eye(n-1)];


%% Question a

% Solve the problems via CVX
% Find x_star1, x_star2

% Solve the l1-regularized problem with CVX
cvx_begin quiet
	variable x(n)
	minimize( 1/2 *  pow_pos(norm(x-d_noised, 2), 2) + lambda * norm(D*x, 1) )
cvx_end

x_star1 = cvx_optpnt.x;
f_opt1 = cvx_optval;

% Solve the l2-regularized problem with CVX
cvx_begin quiet
	variable x(n)
	minimize( 1/2 *  pow_pos(norm(x-d_noised, 2), 2) + lambda * norm(D*x, 2) )
cvx_end

x_star2 = cvx_optpnt.x;

% Figure
figs(1) = figure;
plot(d, '.', 'MarkerSize', markersize);
hold on;
plot(x_star1, '.', 'MarkerSize', markersize);
plot(x_star2, '.', 'MarkerSize', markersize);
plot_setup({'CVX solution for $\ell^1$- and $\ell^2$-regularized problems', sprintf('$\\mathbf{n=%d}$, $\\mathbf{\\lambda = %g}$, $\\mathbf{\\texttt{noise\\_std} = %g}$', n, lambda, noise_std)}, '', '', 'Initial vector', '$\ell^1$-regularized', '$\ell^2$-regularized');


%% Question b

num_of_iters = 100;

% Getting ready to run DPG for One-Dimensional Total Variation Denoising problem (Section 12.4.3 - Algorithm 6)
% Do the proper setup to run the algorithms.

% Initial point
y_0 = randn(n-1, 1);

% Run the DPG algorithm
[x_est_dpg, f_k_dpg] = dpg_1d_tvd(d_noised, lambda, y_0, num_of_iters);


% Getting ready to run FDPG for One-Dimensional Total Variation Denoising problem (Section 12.4.3 - Algorithm 7)
% Do the proper setup to run the algorithms.

% Same initial point as above

% Run the FDPG algorithm
[x_est_fdpg, f_k_fdpg] = fdpg_1d_tvd(d_noised, lambda, y_0, num_of_iters);


% Figure
figs(2) = figure;
plot(d, '.', 'MarkerSize', markersize);
hold on;
plot(x_est_dpg, '.', 'MarkerSize', markersize);
plot(x_est_fdpg, '.', 'MarkerSize', markersize);
plot_setup({'DPG, FDPG solutions for the $\ell^1$-regularized problem', sprintf('$\\mathbf{n=%d}$, $\\mathbf{\\lambda = %g}$, $\\mathbf{\\texttt{noise\\_std} = %g}$, $\\mathbf{\\texttt{num\\_of\\_iters} = %g}$', n, lambda, noise_std, num_of_iters)}, '', '', 'Initial vector', 'DPG', 'FDPG');


% Figure
figs(3) = figure;
semilogy(f_k_dpg - f_opt1);
hold on;
semilogy(f_k_fdpg - f_opt1);
plot_setup({'DPG vs FDPG', sprintf('$\\mathbf{n=%d}$, $\\mathbf{\\lambda = %g}$, $\\mathbf{\\texttt{noise\\_std} = %g}$, $\\mathbf{\\texttt{num\\_of\\_iters} = %g}$', n, lambda, noise_std, num_of_iters)}, '$k$', '$f \left( \mathbf{x}^k \right) - f_{\mathrm{opt}}$', 'DPG', 'FDPG');
