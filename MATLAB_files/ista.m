function [x_est, F_k] = ista(F, f_grad, lambda, L_f, x_init, F_opt, sol_tol)
%ISTA  Perform the ISTA (Iterative-Shrinkage Thresholding Algorithm) on L1
%regularized minimization problem. The problem is 
%min_x {F(x) = f(x) + lambda*norm(x, 1)}  where  f  is convex and  L_f-smooth.
%
%
%USAGE
%
%[x_est, F_k] = ista(F, f_grad, lambda, L_f, x_init, F_opt, sol_tol)
%
%
%PARAMETERS
%
%F : @(x)  where  x : float column (same length as  x_init )
%	The cost function.
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
%	The optimal point of cost function.
%
%sol_tol : float scalar
%	(i.e. solution tolerance) defines the stopping condition as  F(x_k) - F_opt <= sol_tol  which adjusts the accuracy of the solution.
%
%x_est : float column (same length as  x_init )
%	The estimated optimal point.
%
%F_k : float row
%	The value of each general step of the algorithm. The first value is the
%	evaluation of  F  at  x_init . The last value is the estimated optimal value
%	of  F .
%


% Implement the algorithm from Example 10.37


% Initialization
x_k = x_init;

% Store here the  F_k = F(x_k)  values
F_k = F(x_k);

% Start running the algorithm
while ~(F(x_k) - F_opt <= sol_tol)
	x_k = T(lambda/L_f, x_k - 1/L_f * f_grad(x_k));
	F_k = [F_k  F(x_k)]; % store the value
end

x_est = x_k;


end

