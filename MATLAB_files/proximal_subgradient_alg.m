function [x_est, F_k, num_of_iters] = proximal_subgradient_alg(A, b, lambda, x_init, F_opt, sol_tol)
%PROXIMAL_SUBGRADIENT_ALG  Perform the proximal subgradient 
%algorithm on the problem  min_x {F(x) = norm(A*x - b, 1) + lambda*norm(x, 1)} .
%
%
%USAGE
%
%[x_est, F_k, num_of_iters] = proximal_subgradient_alg(A, b, lambda, x_init, F_opt, sol_tol)
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
%lambda : float scalar
%	The coefficient of the l1-norm term.
%
%x_init : float column
%	The initial point.
%
%F_opt : float scalar
%	The optimal point of the cost function (used only in the termination
%	condition).
%
%sol_tol : float scalar
%	(i.e. solution tolerance) defines the stopping condition as
%	F(x_k) - F_opt <= sol_tol  which adjusts the accuracy of the solution.
%
%x_est : float column (same length as  x_init )
%	The estimated optimal point.
%
%F_k : float row
%	The value of each general step of the algorithm. The first value is the
%	evaluation of  F  at  x_init . The last value is the estimated optimal value
%	of  F .
%
%num_of_iters : non-negative integer scalar
%	The number of iterations performed by the algorithm.
%


% Implement the algorithm from Example 9.29


% Assume the functions (as mentioned in the Example 9.29):
% F(x) = norm(A*x-b, 1) + lambda*norm(x, 1)
% f(x) = norm(A*x-b, 1)

F = @(x)  norm(A*x-b, 1) + lambda*norm(x, 1);
f_subgrad = @(x)  l1_norm_aff_subgrad(A, -b, x);

x_k = x_init; % Initialization
F_k = F(x_k); % Store the f_k values here
k = 0; % Iteration counter

% Start running the algorithm
while ~(F(x_k) - F_opt <= sol_tol)
	k = k + 1; % Update the iteration counter
	s_k = dynamic_stepsize(x_k, f_subgrad, k); % Compute the stepsize (dynamic stepsize)
	x_k = T(lambda*s_k, x_k - s_k*f_subgrad(x_k)); % Do the step
	F_k(k+1) = F(x_k); % Store the F_k value
end

x_est = x_k;

num_of_iters = k;

end

