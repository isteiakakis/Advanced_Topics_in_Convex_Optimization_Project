function y = T(lambda, x)
%T  Soft thresholding function with parameter  lambda  evaluated at point  x .
%
%
%USAGE
%
%y = T(lambda, x)
%
%
%PARAMETERS
%
%lambda : float scalar
%	The parameter of the soft thresholding function.
%
%x : float column/row
%	The point at which the evaluation of the soft thresholding function will be
%	made.
%
%y : same type and length as  x
%	The result.
%


% Check out the Example 6.8

y = max(0, abs(x) - lambda) .* sgn(x);

end

