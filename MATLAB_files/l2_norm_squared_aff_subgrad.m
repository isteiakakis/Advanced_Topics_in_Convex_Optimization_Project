function g = l2_norm_squared_aff_subgrad(A, b, x)
%L2_NORM_SQUARED_AFF_SUBGRAD   Compute a weak result of the subgradient of
%norm(A*x+b, 2)  evaluated at point  x .
%
%
%USAGE
%
%g = l2_norm_squared_aff_subgrad(A, b, x)
%
%
%PARAMETERS
%
%A : float matrix
%	If it is a scaled identity matrix, then giving only the scalar value
%	suffices.
%
%b : float column
%	If it is a scaled version of the ones vector (i.e. all its elements are
%	equal), then giving only the scalar value suffices.
%
%x : float column
%
%g : float column (same length as  x )
%	The subgradient of  norm(A*x+b, 2)  evaluated at the given point  x .
%


% The subgradient is just the gradient
g = 2 * A.' * (A*x+b);


end

