function affine_set_projection = project_onto_affine_set(A, b, x)
%PROJECT_ONTO_AFFINE_SET  Compute the projection of point  x  onto the affine
%set  {r in R^n | A*r = b} .
%
%
%USAGE
%
%affine_set_projection = project_onto_affine_set(A, b, x)
%
%
%PARAMETERS
%
%A : float matrix
%	The matrix that describes the affine set, as mentioned above.
%
%b : float column
%	The vector that describes the affine set, as mentioned above.
%
%x : float column
%	The point to be projected onto the described affine set.
%
%affine_set_projection : float column (same length as  x ) 
%	The projection of  x  onto the described affine set.
%


% From Lemma 6.26

affine_set_projection = x - A.' * (A*A.')^-1 * (A*x - b);


end

