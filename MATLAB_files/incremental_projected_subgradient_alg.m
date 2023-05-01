function [x_est, f_k] = incremental_projected_subgradient_alg(f, f_subgrad, f_subgrad_i, projection_func, m, x_init, num_of_iters)
%INCREMENTAL_PROJECTED_SUBGRADIENT_ALG  Perform the incremental subgradient algorithm on
%f = sum_{i=1}^m  f_i .
%
%
%USAGE
%
%[x_est, f_k] = incremental_projected_subgradient_alg(f, f_subgrad, f_subgrad_i, m, x_init, num_of_iters)
%
%
%PARAMETERS
%
%f : @(x)  where  x : float column (same length as  x_init )
%	The cost function (used only to store the general step values, which means
%	that in this case a different function can be provided for evaluation).
%
%f_subgrad : @(x)  where  x : float column (same length as  x_init )
%	A subgradient of  f  evaluated at  x  (used only in the dynamic stepsize
%	evaluation).
%
%f_subgrad_i : @(i, x)  where  i : positive integer scalar , x : float column (same length as  x_init )
%	A subgradient of  f_i  evaluated at  x .
%
%projection_func : @(x)  where  x : float column (same length as  x_init )
%	The function that projects  x  onto the constraint set.
%
%m : positive integer scalar
%	The number of the  f_i  functions.
%
%x_init : float column
%	The initial point.
%
%num_of_iters : positive integer scalar
%	The number of iterations that the algorithm will perform.
%
%x_est : float column (same length as  x_init )
%	The estimated optimal point.
%
%f_k : float row
%	The value of each general step of the algorithm. The first value is the
%	evaluation of  f  at  x_init . The last value is the estimated optimal value
%	of  f .
%


% Implement the algorithm from the section 8.4


% Initialization
x_k = x_init;

% Store the f_k values here
f_k = zeros(num_of_iters+1, 1);
f_k(1) = f(x_k);

for k=1:num_of_iters
	x_ki = x_k;
	t_k = dynamic_stepsize(x_k, f_subgrad, k); % Use the dynamic stepsize

	for i=0:m-1
		x_ki = projection_func(x_ki - t_k * f_subgrad_i(i+1, x_ki));
	end

	x_k = x_ki;
	f_k(k+1) = f(x_k); % Store the f_k value
end

x_est = x_k;


end

