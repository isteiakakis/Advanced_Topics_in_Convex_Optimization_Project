function t_k = polyak_stepsize(x_k, f_subgrad, f, f_opt)
%POLYAK_STEPSIZE  Compute the Polyak stepsize.
%
%
%USAGE
%
%t_k = polyak_stepsize(x_k, f_subgrad, f, f_opt)
%
%
%PARAMETERS
%
%x_k : float column
%	The point of the current general step.
%
%f_subgrad : @(x)  where  x : float column (same length as  x_k )
%	A subgradient of  f  (the cost function) evaluated at  x .
%
%f @(x) : @(x)  where  x : float column (same length as  x_k )
%	The cost function.
%
%f_opt : float scalar
%	The optimal point of the cost function.
%
%t_k : float scalar
%	The Polyak stepsize value.
%


% Formula from the equation (8.13)

f_subgrad_norm = norm(f_subgrad(x_k));

if f_subgrad_norm ~= 0
	t_k = (f(x_k) - f_opt) / f_subgrad_norm^2;
else
	% impossible in practice, but better safe than sorry
	t_k = 1;
end


end
