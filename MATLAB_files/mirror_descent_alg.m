function [x_est, f_k, num_of_iters] = mirror_descent_alg(A, b, x_init, f_opt, sol_tol)
%MIRROR_DESCENT_ALG  Perform the mirror descent algorithm on the problem
%min_{x in unit simplex}  {f(x) = norm(A*x-b, 1)} .
%
%
%USAGE
%
%[x_est, f_k, num_of_iters] = mirror_descent_alg(A, b, x_init, f_opt, sol_tol)
%
%
%PARAMETERS
%
%A : float matrix
%	If it is a scaled identity matrix, then giving only the scalar value
%	suffices.
%
%b : float column
%	If it is a scaled version of the ones vector (i.e. all its elements are
%	equal), then giving only the scalar value suffices.
%
%x_init : float column
%	The initial point.
%
%f_opt : float scalar
%	The optimal point of the cost function (used only in the termination
%	condition).
%
%sol_tol : float scalar
%	(i.e. solution tolerance) defines the stopping condition as
%	f(x_k) - f_opt <= sol_tol  which adjusts the accuracy of the solution.
%
%x_est : float column (same length as  x_init )
%	The estimated optimal point.
%
%f_k : float row
%	The value of each general step of the algorithm. The first value is the
%	evaluation of  f  at  x_init . The last value is the estimated optimal value
%	of  f .
%
%num_of_iters : non-negative integer scalar
%	The number of iterations performed by the algorithm.
%


% Implement the algorithm from Example 9.17


% The cost function
f = @(x)  norm(A*x - b, 1);

% The subgradient function of the cost function
f_subgrad = @(x)  l1_norm_aff_subgrad(A, -b, x);

% Initialization
x_k = x_init;

% Store the f_k values here
f_k(1) = f(x_k);

% Iteration counter
k = 0;

% General step loop
while ~(f(x_k) - f_opt <= sol_tol)
	k = k + 1; % Update the iteration counter
	t_k = dynamic_stepsize(x_k, f_subgrad, k, sqrt(2), inf); % Compute the stepsize
	exp_tk_f_subgrad_xk = exp(-t_k * f_subgrad(x_k));
	x_k = (x_k .* exp_tk_f_subgrad_xk) / (exp_tk_f_subgrad_xk.' * x_k);
	f_k(k+1) = f(x_k); % Store the f_k value
end

x_est = x_k;

num_of_iters = k;


end


