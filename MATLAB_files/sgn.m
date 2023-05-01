function y = sgn(x)
%SGN  Perform the  sgn  function on  x .
%
%
%USAGE
%
%y = sgn(x)
%
%
%PARAMETERS
%
%x : float column
%
%y : float column (same length as  x )
%	If  x  is a scalar, then  y = -1 if x<0 , y = 1 if x>=0 . If  x  is not
%	scalar, then  sgn  is applied elementwise to  x .
%


y = sign(x);
y(y==0) = 1;


end

