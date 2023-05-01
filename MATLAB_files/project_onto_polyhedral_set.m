function [polyhedral_set_projection] = project_onto_polyhedral_set(A, b, x)
%PROJECT_ONTO_POLYHEDRAL_SET  Compute the projection of point  x  onto the
%polyhedral set  {x in R^n | A*x <= b} .
%
%
%USAGE
%
%[polyhedral_set_projection] = project_onto_polyhedral_set(A, b, x)
%
%
%PARAMETERS
%
%A : float matrix
%	The matrix that describes the polyhedral set, as mentioned above.
%
%b : float column
%	The vector that describes the polyhedral set, as mentioned above.
%
%x : float column
%	The point to be projected onto the described polyhedral set.
%
%polyhedral_set_projection : float column (same length as  x )
%	The projection of  x  onto the described polyhedral set.
%


% From section 12.4.1 implement Algorithm 2 (FDPG for this problem)

if all(A*x <= b) == 1
	polyhedral_set_projection = x;
	return;
end

max_iters = 10^6; % maximum number of iterations to perform the general step
tol = 10^-6; % solution tolerance per dimension

% Initialization
[m, ~] = size(A);
L = norm(A, 2)^2; % pick an L such that L >= norm(A, 2)^2
y_init = randn(m, 1);
y_k = y_init;
w_k = y_init;
t_k = 1;

% Start running the algorithm
for k=1:max_iters
	u_k = A.' * w_k + x;
	y_k_next = w_k - 1/L * A * u_k + 1/L * min(A*u_k - L*w_k, b);
	t_k_next = (1+sqrt(1+4*t_k^2))/2;
	w_k_next = y_k_next + (t_k-1)/t_k_next * (y_k_next - y_k);

	if all(A*u_k <= b + tol) == 1
		polyhedral_set_projection = u_k;
		return;
	end

	y_k = y_k_next;
	t_k = t_k_next;
	w_k = w_k_next;
end

warning('Performed the maximum number of iterations without meeting the terminating conditions.');

end

