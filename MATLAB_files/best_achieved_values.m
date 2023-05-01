function v = best_achieved_values(v)
%BEST_ACHIEVED_VALUES  Get the best achieved values of the vector  v .
%
%
%USAGE
%
%v_best = best_achieved_values(v)
%
%
%PARAMETERS
%
%v : float column/row
%	The vector to get the best achieved values.
%
%v_best : same type and length as  v
%	The best achieved values of the vector  v , i.e. for all  i >= 2  we will
%	have that  v_best(i) = min{v_best(i-1), v(i)}  with  v_best(1) = v(1) .
%


for i=2:length(v)
	v(i) = min(v(i-1), v(i));
end


end

