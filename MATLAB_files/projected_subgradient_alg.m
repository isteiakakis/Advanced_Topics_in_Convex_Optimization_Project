function [x_est, f_k, num_of_iters] = projected_subgradient_alg(f, f_subgrad, projection_func, x_init, f_opt, num_of_iters, sol_tol, stepsize_type, dynamic_stepsize_coefficient)
%PROJECTED_SUBGRADIENT_ALG  Perform the projected subgradient algorithm on  f
%which is proper, closed and convex under a constraint set which is nonempty,
%closed and convex and subset of  int(dom(f)).
%
%
%USAGE
%
%[x_est, f_k, num_of_iters_done] = projected_subgradient_alg(f, f_subgrad, projection_func, x_init, f_opt, sol_tol, stepsize_type, dynamic_stepsize_coefficient)
%
%
%PARAMETERS
%
%f : @(x)  where  x : float column (same length as  x_init )
%	The cost function (used to store the value of each general step and in the
%	terminating condition if it exists).
%
%f_subgrad : @(x)  where  x : float column (same length as  x_init )
%	A subgradient of  f  evaluated at  x .
%
%projection_func @(x) : @(x)  where  x : float column (same length as  x_init )
%	The function that projects  x  onto the constraint set.
%
%x_init : float column
%	The initial point.
%
%f_opt : float scalar
%	(Optional, give NaN value if you don't want to provide this argument) The
%	optimal point of the cost function (used in Polyak stepsize, if chosen, and
%	in the terminating condition, see  sol_tol ). If the stepsize is chosen as
%	Polyak, then  f_opt  must be provided.
%
%num_of_iters : positive integer scalar
%	(Optional, give NaN value if you don't want to provide this argument) The
%	number of iterations that the algorithm will perform. If  num_of_iters  is
%	given, then  sol_tol  is ignored.
%
%sol_tol : float scalar
%	(i.e. solution tolerance) Defines the stopping condition as
%	f(x_k) - f_opt <= sol_tol  if  f_opt  is provided, else  sol_tol  is the
%	number of iterations that the algorithm will perform. If  num_of_iters  is
%	given, then sol_tol  is ignored.
%
%stepsize_type : string
%	Choose the stepsize type; either 'polyak' or 'dynamic'.
%
%dynamic_stepsize_coefficient : positive float scalar
%	(Optional, not given or NaN value are equivalent) The coefficient of the
%	stepsize if the dynamic stepsize is chosen.
%
%x_est : float column (same length as  x_init )
%	The estimated optimal point.
%
%f_k : float row
%	The value of each general step of the algorithm. The first value is the
%	evaluation of  f  at  x_init . The last value is the estimated optimal value
%	of  f .
%
%num_of_iters_done : non-negative integer scalar
%	The number of iterations performed by the algorithm.
%


% Implement the algorithm from section 8.2.1


% Choose either Polyak or dynamic stepsize
if strcmp(stepsize_type, 'polyak')
	stepsize_func = @(x_k, f_subgrad, k, f, f_opt)  polyak_stepsize(x_k, f_subgrad, f, f_opt);
elseif strcmp(stepsize_type, 'dynamic')
	if ~exist('dynamic_stepsize_coefficient', 'var') || isnan(dynamic_stepsize_coefficient)
		dynamic_stepsize_coefficient = 1;
	end

	stepsize_func = @(x_k, f_subgrad, k, f, f_opt)  dynamic_stepsize(x_k, f_subgrad, k, dynamic_stepsize_coefficient);
else
	error('stepsize_type argument can take only either ''polyak'' or ''dynamic'' values');
end

% Initialization
x_k = x_init;

% General step loop
if ~isnan(num_of_iters)
	% num_of_iters is given, so do num_of_iters number of iterations

	f_k = zeros(num_of_iters+1, 1);
	f_k(1) = f(x_k); % Store the f_k values here

	for k=1:num_of_iters
		general_step();
		f_k(k+1) = f(x_k); % Store the f_k value
	end
else
	f_k(1) = f(x_k); % Store the f_k values here
	k = 0; % Iteration counter

	while ~(f(x_k) - f_opt <= sol_tol)
		k = k + 1; % Update the iteration counter
		general_step();
		f_k(k+1) = f(x_k); % Store the f_k value
	end

	num_of_iters = k;
end

x_est = x_k;


function general_step()
	t_k = stepsize_func(x_k, f_subgrad, k, f, f_opt); % Compute the stepsize
	x_k = projection_func(x_k - t_k*f_subgrad(x_k)); % Do the step
end


end

