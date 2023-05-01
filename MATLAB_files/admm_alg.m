function [x_est, f_k, num_of_iters] = admm_alg(c, A, b, rho, f_opt, sol_tol)
%ADMM_ALG  Perform the ADMM algorithm on the problem
%min_{x : A*x=b, x>=0} {f(x) = c^T * x} .
%
%
%USAGE
%
%[x_est, f_k] = admm_alg(c, A, b, rho, f_opt, sol_tol)
%
%
%PARAMETERS
%
%c : float column
%
%A : float matrix
%
%b : float column
%
%rho : float scalar
%	The Augmented Lagrangian parameter.
%
%x_init : float column
%	The initial point.
%
%f_opt : float scalar
%	The optimal point of the cost function (used only in the terminating
%	condition, see  sol_tol ).
%
%sol_tol : float scalar
%	(i.e. solution tolerance) Defines the stopping condition as
%	f(x_k) - f_opt <= sol_tol  if  f_opt  is provided, else  sol_tol  is the
%	number of iterations that the algorithm will perform. If  num_of_iters  is
%	given, then sol_tol  is ignored.
%
%x_est : float column (same length as  c )
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


% Implement the algorithm from the report questions 11a, 11b


% The dimension of x
n = length(c);

% The cost function
f = @(x)  c.' * x;

% Initialization
x_k = randn(n, 1);
z_k = randn(n, 1); z_k = abs(z_k);
u_k = randn(n, 1);

% Iteration counter
k = 0;

% Store here the f_k values
f_k(1) = f(x_k);

% Useful results for the inverted KKT matrix
AAA = A.' * (A*A.')^-1;

% Useful results for the general step loop
kkt_inv_matrix_row1 = [eye(n)-AAA*A   AAA];

% General step loop
while ~(abs(f(x_k) - f_opt) <= sol_tol)

	% Perform the general step
	k = k + 1;
	x_k = kkt_inv_matrix_row1 * [rho*z_k-rho*u_k-c ; b];
	z_k = project_onto_box(0, Inf, x_k + u_k);
	u_k = u_k + x_k - z_k;

	% Store the f_k value
	f_k(k+1) = f(x_k);
end

x_est = x_k;

num_of_iters = k;

end


