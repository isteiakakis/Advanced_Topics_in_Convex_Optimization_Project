clear; close all;

% Problem parameters
% m << n
m = 10;
n = 50;

lambda = 10^-3;

% Define the terminating condition; the condition is  f_opt <= c*f(x_k)  iff  f_opt - f(x_k) <= (c-1)*f(x_k)
c = 1.01;


% Generate matrix A
A = randn(m, n);

% Generate sparse vector x_s
s = floor(0.99*m/(2*log(n))); % number of non-zero elements of x_s,  m >= 2*s*log(n) << n
non_zero_indeces = randperm(n);
non_zero_indeces = non_zero_indeces(1:s);

x_s = zeros(n, 1);

for i=non_zero_indeces
	x_s(i) = randn(1);
end

% Compute the vector b
b = A*x_s;

%% Question a
fprintf('Question a\n\n');

% Solve the problems via CVX
% Find x_star1, f_opt1, x_star2, f_opt2

% First problem
fprintf('Solving the L1 regularized problem (CVX)...\n');

cvx_begin quiet
	variable x(n)
	minimize( 1/2 * pow_pos(norm(A*x-b, 2), 2) + lambda * norm(x, 1) )
cvx_end

x_star1 = cvx_optpnt.x;
f_opt1 = cvx_optval;

fprintf('||x_star1 - x_s|| = %f\n\n', norm(x_star1 - x_s));


% Second problem
fprintf('Solving the L2 regularized problem (CVX)...\n');

cvx_begin quiet
	variable x(n)
	minimize( 1/2 * pow_pos(norm(A*x-b, 2), 2) + lambda * pow_pos(norm(x, 2), 2) )
cvx_end

x_star2 = cvx_optpnt.x;
f_opt2 = cvx_optval;

fprintf('||x_star2 - x_s|| = %f\n\n', norm(x_star2 - x_s));


fprintf('Question a: Done\n\n');


%% Question b

% Getting ready to run ISTA and FISTA.
% Do the proper setup to run the algorithms.

fprintf('Question b\n\n');

% The cost function
f1 = @(x)  1/2 * norm(A*x - b)^2 + lambda * norm(x, 1);

% The gradient of the first part of the cost function
f1_part_grad = @(x)  A.' * (A*x - b);

% Initial point for the algorithms
x_0 = randn(n, 1);

% L_f is the maximum eigenvalue of A.'*A or equivalently the square of the maximum singular value of A (Example 5.2)
L_f = sigma_max(A)^2;

% Run ISTA
fprintf('Running ISTA on the L1 regularized problem...\n');
tic
[~, f1_k_ista] = ista(f1, f1_part_grad, lambda, L_f, x_0, f_opt1, (c-1)*f_opt1);
toc

fprintf('\n');

% Run FISTA
fprintf('Running FISTA on the L1 regularized problem...\n');
tic
[~, f1_k_fista] = fista(f1, f1_part_grad, lambda, L_f, x_0, f_opt1, (c-1)*f_opt1);
toc

fprintf('\n');

% Figure
figs(1) = figure;
semilogy(0:length(f1_k_ista)-1, f1_k_ista - f_opt1);
hold on;
semilogy(0:length(f1_k_fista)-1, f1_k_fista - f_opt1);
plot_setup({'ISTA vs FISTA', sprintf('$\\mathbf{(m, n)=(%d, %d)}$, $\\mathbf{\\lambda = %g}$, $\\mathbf{c = %g}$', m, n, lambda, c)}, '$k$', '$f_{\mathrm{best}}^k - f_{\mathrm{opt}}$', 'ISTA', 'FISTA');


fprintf('Question b: Done\n\n');
