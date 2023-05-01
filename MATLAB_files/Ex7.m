clear; close all;

% Problem parameters
m = 300;
n = 320;

% The coefficient of D matrix
rho = 10^-4;

% Solution tolerance (for S-FISTA)
sol_tol = 10^-4;

% Noise standard deviation
noise_std = 0.05;


% Generate x_pwc
num_of_pieces = ceil(n/10); % Create about n/10 number of pieces
x_pwc = zeros(n, 1);
pwc_ending_indeces = [sort(randperm(n-1, num_of_pieces-1)) , n]; % The ending index of each "piece" of x_pwc
from = 1; % The starting index of each piece
for to=pwc_ending_indeces
	x_pwc(from:to) = randn(1, 1); % Create a "piece"
	from = to+1; % Update the starting index for the next piece
end

% Generate A
A = randn(m, n);

% Generate b (with noise)
b = A*x_pwc + noise_std*randn(m, 1);


%% Question a

% Solve the simple LS problem via CVX
% Find x_star1 and f_opt

fprintf('Solving the simple LS problem (CVX)...\n');

cvx_begin quiet
	variable x(n)
	minimize( 1/2 * pow_pos(norm(A*x-b, 2), 2) )
cvx_end

x_star1 = cvx_optpnt.x;
f_opt = cvx_optval;

fprintf('||x_star1 - x_pwc|| = %f\n\n', norm(x_star1 - x_pwc));


%% Question b

% Solve the regularized LS problem via CVX
% Find x_star2 and F_opt

fprintf('Solving the regularized LS problem (CVX)...\n');

% Create the matrix D
D = rho * ([eye(n-1) , zeros(n-1, 1)] + [zeros(n-1, 1) , -eye(n-1)]);

cvx_begin quiet
	variable x(n)
	minimize( 1/2 * pow_pos(norm(A*x-b, 2), 2) + norm(D*x, 1) )
cvx_end

F_opt = cvx_optval;
x_star2 = cvx_optpnt.x;

fprintf('||x_star2 - x_pwc|| = %f\n\n', norm(x_star2 - x_pwc));


%% Question c

% Getting ready to run S-FISTA.
% Do the proper setup to run the algorithm.

% Check out S-FISTA from section 10.8.4.
% Also, check out Example 10.60.
% In this case, S-FISTA is FISTA on the smoothed cost function.

% Cost function
f = @(x)  1/2 * norm(A*x-b, 2)^2;
F = @(x)  f(x) + norm(D*x, 1);

% Initial point for the algorithms
x_0 = randn(n, 1);

% Compute L_tilde
L_f = sigma_max(A)^2;

alpha = sigma_max(D)^2;
beta = n/2;
mu = sqrt(alpha/beta) * sol_tol/(sqrt(alpha*beta) + sqrt(alpha*beta+L_f*sol_tol));

L_tilde = L_f + alpha/mu;

% Gradient of the smoothed cost function
F_smoothed_grad = @(x)  1/2 * l2_norm_squared_aff_subgrad(A, -b, x) + 1/mu * D.' * (D*x-T(mu, D*x));

% Run S-FISTA (which is FISTA on the smoothed cost function and lambda=0); also, store the values f_k (not F_k) and the termination condition is derived from f (not F)
[~, f_k_sfista] = fista(f, F_smoothed_grad, 0, L_tilde, x_0, NaN, NaN, @(x) f(x)-f_opt <= sol_tol);

% Number of S-FISTA iterations.
sfista_num_of_iters = length(f_k_sfista)-1;

% Figure
figs(1) = figure;
semilogy(0:sfista_num_of_iters, f_k_sfista - f_opt);
plot_setup({'S-FISTA', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{\\rho = %g}$, $\\mathbf{\\texttt{sol\\_tol} = %g}$', m, n, rho, sol_tol)}, '$k$', '$f \left( \mathbf{x}^k \right) - f_{\mathrm{opt}}$');


%% Question d

% Getting ready to run the subgradient descent algorithm using dynamic stepsize.
% Do the proper setup to run the algorithm.

% The subgradient of the cost function
F_subgrad = @(x)  1/2 * l2_norm_squared_aff_subgrad(A, -b, x) + l1_norm_aff_subgrad(D, 0, x);

% Run the projected subgradient algorithm without projection using the dynamic stepsize (store the f_k values, not F_k)
[~, f_k_subgrad] = projected_subgradient_alg(f, F_subgrad, @(x) x, x_0, NaN, sfista_num_of_iters, NaN, 'dynamic');

% Figure
figs(2) = figure;
semilogy(0:sfista_num_of_iters, f_k_subgrad - f_opt);
plot_setup({'Subgradient descent', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{\\rho = %g}$, $\\mathbf{\\texttt{sol\\_tol} = %g}$', m, n, rho, sol_tol)}, '$k$', '$f \left( \mathbf{x}^k \right) - f_{\mathrm{opt}}$');


%% Question e

% Figure
figs(3) = figure;
semilogy(0:sfista_num_of_iters, f_k_sfista - f_opt);
hold on;
semilogy(0:sfista_num_of_iters, f_k_subgrad - f_opt);
plot_setup({'S-FISTA vs Subgradient descent', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{\\rho = %g}$, $\\mathbf{\\texttt{sol\\_tol} = %g}$', m, n, rho, sol_tol)}, '$k$', '$f \left( \mathbf{x}^k \right) - f_{\mathrm{opt}}$', 'S-FISTA', 'Subgradient descent');
