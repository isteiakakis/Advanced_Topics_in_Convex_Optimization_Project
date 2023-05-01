function y = sum_huber_scalar_affine_grad(mu, A, b, x)
%SUM_HUBER_SCALAR_AFFINE_GRAD  Compute the gradient of
%sum_{i=1}^m  H_mu (a_i^T * x - b_i)  evaluated at point  x  where  a_i^T  is
%the i-th row of  A ,  b_i  is the i-th element of  b .
%
%
%USAGE
%
%y = sum_huber_scalar_affine_grad(mu, A, b, x)
%
%
%PARAMETERS
%
%mu : float scalar
%	The parameter of the Huber function.
%
%A : float matrix
%
%b : float column
%
%x : float column
%
%y : float scalar
%	The result.
%


% Check out the question 8c in report


% Compute the result of A*x-b
Axb = A*x - b;

y = 1/mu * A.' * (Axb - T(mu, Axb));


end

