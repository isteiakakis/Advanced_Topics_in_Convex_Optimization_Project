function y = H(mu, x)
%H  Huber function with parameter  mu  evaluated at point  x .
%
%
%USAGE
%
%y = H(mu, x)
%
%
%PARAMETERS
%
%mu : float scalar
%	The parameter of the Huber function.
%
%x : float column
%	The point at which the evaluation of the Huber function will be made.
%
%y : float scalar
%	The result.
%


% Check out the equation (6.43)

if norm(x, 2) <= mu
	y = 1/(2*mu)*norm(x, 2)^2;
else
	y = norm(x, 2) - mu/2;
end


end

