function [x_est, F_k] = fista(F, f_grad, lambda, L_f, x_init, F_opt, sol_tol, term_cond, prox_func)
%FISTA  Perform the FISTA (Fast Iterative-Shrinkage Thresholding
%Algorithm) on L1 regularized minimization problem. The problem is
%min_x {F(x) = f(x) + lambda*norm(x, 1)}  where  f  is convex and
%L_f-smooth.
%
%
%USAGE
%
%[x_est, F_k] = fista(F, f_grad, lambda, L_f, x_init, F_opt, sol_tol, term_cond, prox_func)
%
%
%PARAMETERS
%
%F : @(x)  where  x : float column (same length as  x_init)
%	The cost function (used to store the value of each general step and in the
%	terminating condition if  term_cond  is not provided; if  term_cond  is
%	provided, then it is ONLY used to store the general step values, which means
%	that in this case a different function can be provided for evaluation).
%
%f_grad : @(x)  where  x : float column (same length as  x_init )
%	The gradient of f.
%
%lambda : float scalar
%	The coefficient of the l1-norm term.
%
%L_f : float scalar
%	The smooth parameter of f, i.e. f is L_f-smooth.
%
%x_init : float column
%	The initial point.
%
%F_opt : float scalar
%	The optimal point of the cost function (used only in the default termination
%	condition).
%
%sol_tol : float scalar
%	(i.e. solution tolerance) defines the stopping condition as
%	F(x_k) - F_opt <= sol_tol  which adjusts the accuracy of the solution.
%
%term_cond : @(x)  where  x : float column (same length as  x_init )
%	(Optional, not given or NaN value are equivalent) Redefines the terminating
%	condition (if given, F_opt and sol_tol are ignored; if not given then, the
%	default condition is the one mentioned above).
%
%prox_func : @(x)  where  x : float column (same length as  x_init )
%	(Optional, not given or NaN value are equivalent) Redefines the function in
%	the step using the promimal operator (if not given, the default function is
%	the soft thresholding function; if given, lambda and L_f are not used).
%
%x_est : float column (same length as  x_init )
%	The estimated optimal point.
%
%F_k : float row
%	The value of each general step of the algorithm. The first value is the
%	evaluation of  F  at  x_init . The last value is the estimated optimal value
%	of  F .
%


% Do the proper setup according to the given arguments

if ~exist('term_cond', 'var') || ~isa(term_cond, 'function_handle')
	% Use the default terminating condition
	term_cond = @(x)  F(x) - F_opt <= sol_tol;
end

if ~exist('prox_func', 'var') || ~isa(prox_func, 'function_handle')
	% Use the default function in the proximal step, which is the soft-thresholding operator
	prox_func = @(input)  T(lambda/L_f, input);
end


% Implement the algorithm from Example 10.37


% Initialization
x_k = x_init;
y_k = x_init;
t_k = 1;

% Store here the  F_k = F(x_k)  values for each k (i.e. for each iteration)
F_k = F(x_k);

% Start running the algorithm
while ~term_cond(x_k)

	% General step
	x_k_next = prox_func(y_k - 1/L_f * f_grad(y_k));
	t_k_next = (1 + sqrt(1+4*t_k^2))/2;
	y_k = x_k_next + (t_k-1)/t_k_next * (x_k_next - x_k);

	% Update the values for the next iteration
	x_k = x_k_next;
	t_k = t_k_next;

	% Store the value
	F_k = [F_k  F(x_k_next)];
end

x_est = x_k;


end

