function [x_est, f_k] = fdpg_1d_tvd(d, lambda, y_init, num_of_iters)
%FDPG  Perform the FDPG (Fast Dual Proximal Gradient) algorithm on the
%One-Dimensional Total Variation Denoising problem which is
%min_x {1/2 * norm(x-d, 2)^2 + lambda * sum_{i=1}^{n-1}  |x_i-x_{i+1}|} .
%
%
%USAGE
%
%[x_est, f_k] = fdpg_1d_tvd(d, lambda, y_init, num_of_iters)
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


% Implement the algorithm 7 from Section 12.4.3


% The length of  u_k
n = length(y_init) + 1;

% The cost function
f = @(x)  1/2 * norm(x-d)^2 + lambda * norm(x(1:n-1) - x(2:n), 1);

% Store here the f_k values
f_k = zeros(num_of_iters, 1);

% Initialization
y_k = y_init;
w_k = y_init;
t_k = 1;

% Start running the algorithm
for i=1:num_of_iters

	% General step
	u_k = [w_k ; 0] - [0 ; w_k] + d;
	y_k_next = w_k - 1/4 * (u_k(1:n-1) - u_k(2:n)) + 1/4 * T(4*lambda, (u_k(1:n-1) - u_k(2:n)) - 4*w_k);
	t_k_next = (1 + sqrt(1+4*t_k^2))/2;
	w_k = y_k_next + (t_k-1)/t_k_next * (y_k_next - y_k);

	% Update the y_k, t_k values for the next iteration
	y_k = y_k_next;
	t_k = t_k_next;

	% Store the f_k value
	f_k(i) = f(u_k);
end

x_est = u_k;


end

