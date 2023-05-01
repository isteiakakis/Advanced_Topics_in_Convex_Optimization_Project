clear; close all;

% m <= n (because A is full row rank in order for the problem to be feasible)
m = 1500;
n = 2000;

% Generate a problem that is feasible, i.e. S = intersection(S_1, S_2) is non-empty
A = randn(m, n);
b = A * abs(randn(n, 1)); % generate b this way so as the problem is feasible

% Create the function that projects onto R_+ which is a special case of the projection onto box
project_onto_Rplus = @(x)  project_onto_box(0, inf, x);


%% Implement the Algorithm 1 from the Example 8.23
% Perform alternating projection: firstly onto the affine set, and then onto R_+

%===========================================================================

% Solution tolerance (for the affine set)
sol_tol = 10^-3;

% Initialization
x_k = ones(n, 1);

while ~( norm(A*x_k - b) <= sol_tol && min(x_k) >= 0 )
	% Do the alternating projection
	x_k = project_onto_Rplus(project_onto_affine_set(A, b, x_k));
end

x_est = x_k;

%===========================================================================

fprintf('Solution found (in  x_est  variable).\n\n');


