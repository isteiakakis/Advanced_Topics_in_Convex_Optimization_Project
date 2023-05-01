function [x_est, f_k] = dpg_1d_tvd(d, lambda, y_init, num_of_iters)
%DPG  Perform the DPG (Dual Proximal Gradient) algorithm on the One-Dimensional
%Total Variation Denoising problem which is
%min_x {1/2 * norm(x-d, 2)^2 + lambda * sum_{i=1}^{n-1}  |x_i-x_{i+1}|} .
%
%
%USAGE
%
%[x_est, f_k] = dpg_1d_tvd(d, lambda, y_init, num_of_iters)
%
%
%PARAMETERS
%
%d : float column
%	The noisy vector of the problem.
%
%lambda : float scalar
%	The coefficient of the l1-norm term.
%
%y_init : float column
%	The initial point.
%
%num_of_iters
%	The number of iterations that the algorithm will perform.
%
%x_est : float column (its length is larger by one than  y_init )
%	The estimated optimal point.
%
%f_k : float row
%	The value of each general step of the algorithm. The last value is the
%	estimated optimal value of  f .
%


% Implement the algorithm 6 from Section 12.4.3


% The length of  x_k
n = length(y_init) + 1;

% The cost function
f = @(x)  1/2 * norm(x-d)^2 + lambda * norm(x(1:n-1) - x(2:n), 1);

% Store here the f_k values
f_k = zeros(num_of_iters, 1);

% Initialization
y_k = y_init;

% Start running the algorithm
for i=1:num_of_iters

	% General step
	x_k = [y_k ; 0] - [0 ; y_k] + d;
	y_k = y_k - 1/4 * (x_k(1:n-1) - x_k(2:n)) + 1/4 * T(4*lambda, (x_k(1:n-1) - x_k(2:n)) - 4*y_k);

	% Store the f_k value
	f_k(i) = f(x_k);
end

x_est = x_k;


end

