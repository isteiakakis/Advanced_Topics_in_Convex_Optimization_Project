function [x_est, f_k] = stochastic_projected_subgradient_alg(f, f_subgrad_i, projection_func, L_f_i, m, theta, x_init, num_of_epochs)
%STOCHASTIC_PROJECTED_SUBGRADIENT_ALG  Perform the stochastic projected
%subgradient descent algorithm on  f  under a constraint set which is nonempty,
%closed, convex and compact.
%
%
%USAGE
%
%[x_est, f_k] = stochastic_subgradient_alg(f, f_subgrad_i, L_f_i, m, theta, x_init, num_of_epochs)
%
%
%PARAMETERS
%
%f : @(x)  where  x : float column (same length as  x_init )
%	The cost function (used only to store the general step values, which means
%	that in this case a different function can be provided for evaluation).
%
%f_subgrad_i : @(i, x)  where  i : positive integer scalar , x : float column (same length as  x_init )
%	A subgradient of  f_i  evaluated at  x .
%
%projection_func : @(x)  where  x : float column (same length as  x_init )
%	The function that projects  x  onto the constraint set.
%
%L_f_i : float column/row
%	Each element of the column/row  L_f_i  is a Lipschitz constant of  f_i .
%
%m : positive integer scalar
%	The number of the  f_i  functions.
%
%theta : float scalar
%	Some upper bound on the half-squared diameter of the constraint set.
%
%x_init : float column
%	The initial point.
%
%num_of_epochs : positive integer scalar
%	The number of epochs that the algorithm will perform.
%
%x_est : float column (same length as  x_init )
%	The estimated optimal point.
%
%f_k : float row
%	The value of each general step of the algorithm. The first value is the
%	evaluation of  f  at  x_init . The last value is the estimated optimal value
%	of  f .
%


% Implement algorithm 2 from the Example 8.36

L_f_tilde = sqrt(m * sum(L_f_i)^2);

x_k = x_init; % Initialization

% Store the f_k values here
f_k = zeros(num_of_epochs+1, 1);
f_k(1) = f(x_k);

% Start running the algorithm
for k=1:num_of_epochs
	for p=1:m % one epoch
		i_k = randi(m, 1);

		t_k = dynamic_stepsize(x_k, @(x) L_f_tilde, (k-1)*m+p, sqrt(2*theta)*m, 1); % here the correct iteration counter is equal to (k-1)*m+p
		x_k = projection_func(x_k - t_k * f_subgrad_i(i_k, x_k));
	end

	f_k(k+1) = f(x_k); % Store the f_k value
end

x_est = x_k;


end

