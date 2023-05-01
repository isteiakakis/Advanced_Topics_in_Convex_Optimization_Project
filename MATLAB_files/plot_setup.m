function plot_setup(plot_title, plot_xlabel, plot_ylabel, varargin)
%PLOT_SETUP  Util function to set the title, the axes labels and the legends of
%the current plot.
%
%
%USAGE
%
%plot_setup(plot_title, plot_xlabel, plot_ylabel, ...)
%
%
%PARAMETERS
%
%plot_title : string
%	The title of the plot. If it is the empty string, then it is ignored.
%
%plot_xlabel : string
%	The label of x axis of the plot. If it is the empty string, then it is
%	ignored.
%
%plot_ylabel : string
%	The label of y axis of the plot. If it is the empty string then it is
%	ignored.
%
%... : strings
%	(Optional) The rest arguments are the legends of the plot.
%


grid on;
set(gca, 'FontSize', 11);

% Set up the title
if ~isempty(plot_title)
	if ~iscell(plot_title)
		plot_title = sprintf('\\textbf{%s}', plot_title);
	else
		for i=1:length(plot_title)
			plot_title{i} = sprintf('\\textbf{%s}', plot_title{i});
		end
	end

	title(plot_title, 'HorizontalAlignment', 'center', 'Interpreter', 'latex', 'FontSize', 15);
end

% Set up the x label
if ~isempty(plot_xlabel)
	xlabel(plot_xlabel, 'Interpreter', 'latex', 'FontSize', 15);
end

% Set up the y label
if ~isempty(plot_ylabel)
	ylabel(plot_ylabel, 'Interpreter', 'latex', 'FontSize', 15);
end

% Set up the legends
if nargin > 3
	legend_handle = legend(varargin{:}, 'Location', 'Best', 'Interpreter', 'latex', 'FontSize', 11);
end


end

