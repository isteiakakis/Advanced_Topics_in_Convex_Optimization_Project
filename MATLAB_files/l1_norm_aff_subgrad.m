function g = l1_norm_aff_subgrad(A, b, x)
%L1_NORM_AFF_SUBGRAD  Compute a weak result of the subgradient of
%norm(A*x+b, 1)  evaluated at point  x .
%
%
%USAGE
%
%g = l1_norm_aff_subgrad(A, b, x)
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
%	The subgradient of  norm(A*x+b, 1)  evaluated at the given point  x .
%


% Check out Example 3.44

g = A.' * sgn(A*x + b);


end

