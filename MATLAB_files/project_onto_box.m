function box_projection = project_onto_box(l, u, x)
%PROJECT_ONTO_BOX  Compute the projection of point  x  onto the box set where
%box[l, u] = {r in R^n | l <= r <= u} .
%
%
%USAGE
%
%box_projection = project_onto_box(l, u, x)
%
%
%PARAMETERS
%
%l : float/inf column (same length as  x )
%	The lower bound of the box set.
%
%u : float/inf column (same length as  x )
%	The upper bound of the box set.
%
%x : float column
%	The point to be projected onto the described box set.
%
%box_projection : float column (same length as  x )
%	The projection of  x  onto the described box set.
%


% Check out Lemma 6.26

box_projection = min(max(x, l), u);


end

