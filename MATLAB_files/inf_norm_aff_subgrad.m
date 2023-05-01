function g = inf_norm_aff_subgrad(A, b, x)
%INF_NORM_AFF_SUBGRAD  Compute a weak result of the subgradient of
%norm(A*x+b, inf)  evaluated at point  x .
%
%
%USAGE
%
%g = inf_norm_aff_subgrad(A, b, x)
%
%
%PARAMETERS
%
%A : float matrix
%
%b : float column
%
%x : float column
%
%g : float column (same length as  x )
%	The subgradient of  norm(A*x+b, inf)  evaluated at the given point  x .
%


% Check out Example 3.54


% Find the indeces where the value equals the infinity norm of A*x+b
I = find( abs(A*x+b) == norm(A*x+b, Inf) );
I = I.'; % make it a row vector in order to iterate over its every element in the for loop bellow

% Choose, arbitrarily, lambda_i to be equal for all i in I
lambda_i = 1/length(I);

g = 0;

for i=I
	g = g + sgn(A(i,:) * x + b(i)) * A(i,:).';
end

g = lambda_i * g;


end

