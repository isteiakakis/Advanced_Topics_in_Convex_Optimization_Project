function unit_simplex_projection = project_onto_unit_simplex(x)
%PROJECT_ONTO_UNIT_SIMPLEX  Compute the projection of point  x  onto the unit
%simplex.
%
%
%USAGE
%
%unit_simplex_projection = project_onto_unit_simplex(x)
%
%
%PARAMETERS
%
%x : float column
%	The point to be projected onto the unit simplex set.
%
%unit_simplex_projection : float column (same length as  x )
%	The projection of  x  onto the unit simplex set.
%


% Check out Corollary 6.29


% The projection function (needs mu_star)
project_onto_unit_simplex_mu = @(x, mu)  max(x - mu, 0);

% Evaluate the equation whose mu_star is a root
eval_eq = @(x, mu)  sum(project_onto_unit_simplex_mu(x, mu)) - 1;

% Find a root mu_star of the equation
% Use the bisection method
% (mu_1, mu_2) is the interval of the solution
% mu_1, mu_2 must evaluate the equation to results with different sign

% Arbitrary initialization
mu_1 = -1000;
mu_2 = 1000;

% Make mu_1 small enough so as to evaluate the equation to negative result
while eval_eq(x, mu_1) < 0
	mu_1 = mu_1 - 1000;
end

% Make mu_2 big enough so as to evaluate the equation to positive result
while eval_eq(x, mu_2) > 0
	mu_2 = mu_2 + 1000;
end

% Find the middle of the interval
mid = (mu_1 + mu_2) / 2;

% Evaluate the middle
equation_mid_val = eval_eq(x, mid);

% Apply the bisection at each iteration until the result is close enough to
% the real one
while abs(equation_mid_val) > 10^-6
	if equation_mid_val > 0
		mu_1 = mid;
	else
		mu_2 = mid;
	end

	mid = (mu_1 + mu_2) / 2;
	equation_mid_val = eval_eq(x, mid);
end

% The root
mu_star = mid;

% Now the root has been found, apply the projection
unit_simplex_projection = project_onto_unit_simplex_mu(x, mu_star);


end

