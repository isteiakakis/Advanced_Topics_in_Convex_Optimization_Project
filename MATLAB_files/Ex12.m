clear; close all;

% Problem parameters
% m <= n because A must be full row rank so as the constraint A*x=b has feasible points
m = 200;
n = 500;


%% Question a

fprintf('Question a\n');

% Generate a non empty S_1
x_true = randn(n, 1);
A = randn(m, n);
b = A*x_true + exprnd(1, m, 1);

% The point that is going to be projected 
d = randn(n, 1);

% Perform the projection
projection_onto_S_1 = project_onto_polyhedral_set(A, b, d);

% Solution found
fprintf('Solution found (in  projection_onto_S_1  variable).\n\n');


%% Question b

fprintf('Question b\n');

% Generate a non empty S_2
x_true = randn(n, 1);
A = randn(m, n);
b = A*x_true;
v_1 = x_true - exprnd(1, n, 1);
v_2 = x_true + exprnd(1, n, 1);

% The point that is going to be projected 
d = randn(n, 1);

% Perform the projection
projection_onto_S_2 = project_onto_intersection(d, ...
	{@(x) project_onto_affine_set(A, b, x) , @(x, tol) all(abs(A*x - b) <= tol)}, ...
	{@(x) project_onto_box(v_1, v_2, x)    , @(x, tol) all(v_1 - tol <= x) && all(x <= v_2 + tol)} );

% Solution found
fprintf('Solution found (in  projection_onto_S_2  variable).\n\n');


%% Question c

fprintf('Question c\n');

% Generate a non empty S_3
x_true = randn(n, 1);
A = randn(m, n);
b = A*x_true + exprnd(1, m, 1);
v_1 = x_true - exprnd(1, n, 1);
v_2 = x_true + exprnd(1, n, 1);

% The point that is going to be projected 
d = randn(n, 1);

% Perform the projection
projection_onto_S_3 = project_onto_intersection(d, ...
	{@(x) project_onto_polyhedral_set(A, b, x) , @(x, tol) all(A*x <= b + tol)}, ...
	{@(x) project_onto_box(v_1, v_2, x)        , @(x, tol) all(v_1 - tol <= x) && all(x <= v_2 + tol)} );

% Solution found
fprintf('Solution found (in  projection_onto_S_3  variable).\n\n');


