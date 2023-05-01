clear; close all;

% Dimension number
n = 2;
%n = 10^8; % In this case, our algorithm runs in about 8 seconds (ATTENTION: in such high-dimensional cases, CVX is too slow, so in those cases the script returns before CVX)

% The point that will be projected (randomly generated: all the elements are iid, normal distribution with standard deviation 3)
x = 3*randn(n, 1);

% Run the algorithm to compute the projection
unit_simplex_projection = project_onto_unit_simplex(x);

% Solution found
fprintf('Solution found (in  unit_simplex_projection  variable).\n\n');

% In very high-dimensional cases CVX is too slow, don't run it (unless you are willing to wait)
if n > 10^5
	return;
end

% Check with CVX
cvx_begin quiet
	variable z(n)
	minimize( norm(x-z, 2) )
	subject to
		z >= 0
		sum(z) == 1
cvx_end

% Print the norm of the difference of found projection and CVX projection
compared_to_cvx_error = norm(cvx_optpnt.z - unit_simplex_projection);
compared_to_cvx_error


% Plot the projection for n=2
if n==2
	% Figure
	figs(1) = figure;
	hold on;

	% Add the plots
	plot([1 0],[0 1]); % Unit simplex
	plot([x(1) unit_simplex_projection(1)], [x(2) unit_simplex_projection(2)], '--'); % The projection dashed line
	plot(x(1), x(2), 'k.', 'MarkerSize', 10); % The point that is going to be projected
	plot(unit_simplex_projection(1), unit_simplex_projection(2), 'k.', 'MarkerSize', 10); % The projected point
	plot_setup('Projection onto Unit Simplex', '', '', 'Unit Simplex', 'Projection'); % Title, legends
	axis equal;

	% Add the description in each point
	text(x(1), x(2), 'x', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', 'FontSize', 14, 'FontWeight', 'bold');
	text(unit_simplex_projection(1), unit_simplex_projection(2), 'x_*', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', 'FontSize', 14, 'FontWeight', 'bold');
end

