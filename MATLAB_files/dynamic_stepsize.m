function t_k = dynamic_stepsize(x_k, f_subgrad, k, coefficient, pnorm)
%DYNAMIC_STEPSIZE  Compute the dynamic stepsize.
%
%
%USAGE
%
%t_k = dynamic_stepsize(x_k, f_subgrad, k, coefficient, pnorm)
%
%
%PARAMETERS
%
%x_k : float column
%	The point of the current general step.
%
%f_subgrad : @(x)  where  x : float column (same length as  x_k )
%	A subgradient of  f  evaluated at  x .
%
%k : non-negative integer scalar
%	The current iteration counter.
%
%coefficient : float scalar
%	(Optional, not given or NaN value are equivalent) A coefficient that scales
%	the returning dynamic stepsize.
%
%pnorm : positive integer or Inf scalar
%	(Optional, not given or NaN value are equivalent) The norm that will be used
%	for the subgradient of f.
%
%t_k : float scalar
%	The dynamic stepsize value.
%


% Do the proper setup according to the given arguments

if ~exist('coefficient', 'var') || isnan(coefficient)
	coefficient = 1;
end

if ~exist('pnorm', 'var') || isnan(pnorm)
	pnorm = 2;
end


% Formula from Theorem 8.28

f_subgrad_pnorm = norm(f_subgrad(x_k), pnorm);

if f_subgrad_pnorm ~= 0
	t_k = coefficient * 1/(f_subgrad_pnorm * sqrt(k+1));
else
	% impossible in practice, but better safe than sorry
	t_k = coefficient;
end


end
