function [intersection_projection] = project_onto_intersection(x, varargin)
%PROJECT_ONTO_INTERSECTION  Compute the projection of x onto the intersection
%of closed convex sets.
%
%USAGE
%
%[intersection_projection] = project_onto_intersection(x, ...)
%
%
%PARAMETERS
%
%x : float column
%	The point to be projected onto the described intersetion of sets.
%
%... : cells
%	Each one of the rest arguments corresponds to each set of the intersected
%	sets. Each one (of the rest arguments) is a cell with two elements:
%	- A projection function  @(x)  that projects  x  onto each set and returns
%	  its projection.
%	- A testing function  @(x, tol)  that checks if  x  belongs to
%	  each set with an elementwise tolerance  tol  (positive float scalar) and
%	  returns  1  if it does, else  0 .
%
%intersection_projection
%	The projection of  x  onto the described intersetion of sets.
%


% From section 12.4.2 implement Algorithm 4 (FDPG for this problem)

% The function of projection and termination checking
project__terminate = varargin;

if termination_condition(x, 0)
	intersection_projection = x;
	return;
end

max_iters = 10^6; % maximum number of iterations to perform the general step
tol = 10^-5; % solution tolerance per dimension

p = nargin - 1; % the number of arguments in  project__terminate , which is the number of the sets of the intersection
n = length(x);

% Initialization
L = p; % pick an L such that L >= p  where p is the number of the convex sets in intersection
y_init = randn(n, p);
y_k = y_init;
w_k = y_init;
t_k = 1;

y_k_next = zeros(size(y_k)); % preallocation

% Start running the algorithm
for k=1:max_iters
	u_k = sum(w_k, 2);

	for i=1:p
		y_k_next(:, i) = w_k(:, i) - 1/L * u_k + 1/L * project__terminate{i}{1}(u_k - L*w_k(:, i)); % step with projection onto each set
	end

	t_k_next = (1+sqrt(1+4*t_k^2))/2;

	w_k_next = y_k_next + (t_k-1)/t_k_next * (y_k_next - y_k);

	if termination_condition(u_k, tol)
		intersection_projection = u_k;
		return;
	end

	% Update the iteration values for the next loop
	y_k = y_k_next;
	t_k = t_k_next;
	w_k = w_k_next;
end

warning('Performed the maximum number of iterations without meeting the terminating conditions.');


function bool_terminate = termination_condition(x, tol)
	% if all the termination functions return 1 then bool_terminate = 1, else bool_terminate = 0
	for j=1:length(project__terminate)
		if project__terminate{j}{2}(x, tol) == 0
			bool_terminate = 0;
			return;
		end
	end

	bool_terminate = 1;
end


end
