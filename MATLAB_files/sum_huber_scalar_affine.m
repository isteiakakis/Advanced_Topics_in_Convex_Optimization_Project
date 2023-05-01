function y = sum_huber_scalar_affine(mu, A, b, x)
%SUM_HUBER_SCALAR_AFFINE  Compute the  sum_{i=1}^m  H_mu (a_i^T * x - b_i)
%where  a_i  is the i-th row of  A ,  b_i  is the i-th element of  b .
%
%
%USAGE
%
%y = sum_huber_scalar_affine(mu, A, b, x)
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


% Number of rows of A
m = size(A, 1);

% Compute the return value
y = 0;

for i=1:m
	y = y + H(mu, A(i, :) * x - b(i));
end


end


