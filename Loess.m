classdef Loess < handle
% LOESS performs a local weighted robust regression in N-dimensions
%
%   Loess properties:
%       points              - Data points
%       values              - Data values
%       span                - Span or window for the local regression
%       order               - Order of local regression
%       robust_iterations   - Number of robust iterations
%       n_in_span           - Number of points in span (read only)
%       n_dims              - Number of dimensions (read only)
%       n_points            - Number of points (read only)
%
%   Loess methods:
%       Loess               - Construct a Loess object
%       interp              - Performs inter(extra)-polation
%
%   Example:
%         % Some parameters
%         n_inputs=10000; % Number of input points
%         n_outliers=100; % Number of outliers in input
%         n_nans=0;       % Number of NaNs in input
%         [XI,YI]=meshgrid(linspace(-1, 1, 100), linspace(-1,1,100)); % Mesh to interpolate to
%
%         % Create input data
%         x=[1-2*rand(n_inputs,1) 1-2*rand(n_inputs,1)];
%         val=sin(pi*x(:,1).*x(:,2))/0.5+x(:,1).^2-exp(x(:,1));
%         val(ceil(length(x)*rand(n_outliers,1))) = 15-30*rand(n_outliers,1);
%         val(ceil(length(x)*rand(n_nans,1))) = nan;
%
%         % Create Loess object
%         L=Loess;
%         L.order=2;             % Set interpolation order
%         L.robust_iterations=2; % Set number of robust iterations
%         L.span=0.01;           % Set filter window
%         L.points=x;            % Set input points
%         L.values=val;          % Set input values
%
%         % Perform the interpolation
%         VI=reshape( L.interp([XI(:),YI(:)]), size(XI) );
%
%         % Plot the result
%         close all
%         plot3(x(:,1),x(:,2),val,'k.')
%         hold on
%         surf(XI,YI,VI)
%         camlight('right')
%         shading flat
%         set(gca,'projection','perspective')
%         colorbar
%         legend('Input points','Fitted values','location','NorthEast')

% The MIT License (MIT)
% 
% Copyright (c) 2016 Bart Vermeulen
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

%   Note that the Loess object relies on the statistics toolbox to build
%   perform a kNN search. Failure to checkout the statistics toolbox will
%   result in an error upon object construction.

%   If the paralled toolbox is available the local regression is performed
%   in parallel

    properties(SetObservable, AbortSet)
        % points defines the input points. It is an NxM matrix with N being
        % the number of input points and M the number of dimensions.
        % Default is 0x1 matrix
        points
    end

    properties
        % values at the input points. It is an Nx1 vector with N being
        % equal to the number of input points. Default is 0x1 matrix.
        values

        % size of the window in which the regression is performed. It
        % is a scalar value which can be a value between 0 and 1 to define
        % the fraction of points to include in the regression, or a number
        % higher than 1 to define the number of points to include in the
        % regression. The actual value used for the regression is stored in
        % the property n_in_span. Default is 0.2.
        span

        % Order defines the order of the local regression. Order can be
        % equal to 1 or 2. Default is 1.
        order

        % Number of robust iterations to remove outliers. Default is 0.
        robust_iterations
    end

    properties(Dependent, GetAccess=public,SetAccess=protected)
        valid_points

        valid_values

        n_valid_points

        % Actual number of points included in the local regression. Depends
        % on the number of input points, the span property and the order
        % property
        n_in_span

        % Number of dimensions. Returns the number of columns of the points
        % property
        n_dims

        % Number of points. Returns the number of rows of the points
        % property
        n_points
    end

    properties(Hidden,Access=private)
        % kdtree to perform kNN-search
        kdtree;

        % if true kdtree needs to be rebuilt (triggered when input points
        % are changed)
        rebuild_kdtree;
    end

    methods
        % CONSTRUCTOR
        function obj=Loess(varargin)
            % Loess constructs a Loess object
            %
            %   Loess() Constructs an empty Loess object
            %
            %   Loess(X,V) sets the input points X and the corresponding
            %       values V. X is an NxM matrix with N being the number of
            %       points and M the number of dimensions. V is an Nx1
            %       vector.
            %
            %   Loess(...,'propertyName',propertyValue') allows to set the
            %       object properties upon construction

            % check availability of the statistics toolbox
            assert(license('checkout','statistics_toolbox')==1, 'Unable to checkout the statistics toolbox')

            % Create inputParser and define input
            P=inputParser;
            P.FunctionName='Loess';
            P.addOptional('points',double.empty(0,1))
            P.addOptional('values',double.empty(0,1))
            P.addParameter('span',0.2)
            P.addParameter('order',1)
            P.addParameter('robust_iterations',0)

            % Parse input
            P.parse(varargin{:})

            % Assign parsed input to object properties
            obj.points=P.Results.points;
            obj.values=P.Results.values;
            obj.span=P.Results.span;
            obj.order=P.Results.order;
            obj.robust_iterations=P.Results.robust_iterations;

            % Initialize kdtree and rebuild flag
            obj.kdtree=[];
            obj.rebuild_kdtree=false;

            % Add listener for changes to the points property
            addlistener(obj,'points','PostSet',@obj.reset_kdtree);
        end

        %%% SET AND GET METHODS
        function val=get.valid_points(obj)
            assert(size(obj.points,1)==size(obj.values,1),'points and values properties should have the same number of rows')
            val=obj.points(all(isfinite(obj.points),2) & isfinite(obj.values),:);
        end

        function val=get.valid_values(obj)
            assert(size(obj.points,1)==size(obj.values,1),'points and values properties should have the same number of rows')
            val=obj.values(all(isfinite(obj.points),2) & isfinite(obj.values));
        end

        function n=get.n_valid_points(obj)
            n=size(obj.valid_points,1);
        end

        function val=get.n_in_span(obj)
            if obj.span > 1
                val=min(max(obj.order+1,obj.span),size(obj.points,1));
            else
                val=min(max(obj.order+1,round(size(obj.points,1)*obj.span)),size(obj.points,1));
            end
        end

        function val=get.n_dims(obj)
            val=size(obj.points,2);
        end

        function val=get.n_points(obj)
            val=size(obj.points,1);
        end

        function set.points(obj,val)
            validateattributes(val,{'single','double'},{'2d'})
            obj.points=val;
        end

        function set.values(obj,val)
            validateattributes(val,{'single','double'},{'column'})
            obj.values=val;
        end

        function set.order(obj,val)
            validateattributes(val,{'numeric'},{'integer','scalar','>=',1,'<=',2})
            obj.order=val;
        end

        function set.span(obj,val)
            validateattributes(val,{'numeric'},{'finite','scalar','>=',0})
            obj.span=val;
        end

        %%% GENERIC METHODS
        function query_values=interp(obj,varargin)
            % interp interpolates values to given query points
            %
            %   query_values = interp(Loess_object) returns the local
            %   fitted values at the input locations. This acts as a
            %   smoother of the data values at the input locations.
            %
            %   query_values = interp(Loess_object, query_points)
            %   interpolates input values to the given query_points.
            %   query_points is an NxM matrix with N number of points and M
            %   number of dimensions. The number of dimensions should match
            %   the number of dimensions of input points
            %
            %   See also: Loess

            % Check input and properties are ok
            if obj.n_valid_points<obj.order+1
                error('%d points are not enough for regression of order %d',obj.n_valid_points,obj.order)
            end

            if nargin < 2
                query_points=obj.points;
            else
                query_points=varargin{1};
                assert(size(query_points,2)==obj.n_dims,'points property and query_points should have same number of columns')
            end

            % Start robust iterations (These iterations are performed with
            %   the input points as query points)
            robust_weights = ones(obj.n_valid_points,1);                          % Initialize vector to hold robust values
            for count_robust_iter=1:obj.robust_iterations                   % Loop over the given number of robust iterations
                query_values = obj.local_fit(obj.valid_points, robust_weights);   % Perform first local fit
                query_residuals = query_values-obj.valid_values;                  % Compute residuals of the fit
                x_bisquare=query_residuals/(6*median(abs(query_residuals)));% Compute argument of bisquare function
                robust_weights = (1 - x_bisquare .^ 2) .^ 2;                % Compute robust weights
                robust_weights(abs(x_bisquare) >= 1) = 0;                   % Compute robust weights
            end

            % Compute final interpolation
            fgood=all(isfinite(query_points),2);                            % Check for nans in query_points coordinates
            query_values=nan(size(query_points,1),1);                       % Initialize output values
            query_values(fgood) = obj.local_fit(query_points(fgood,:), robust_weights); % Perform final fit
        end
    end

    methods (Hidden,Access = protected)
        function varargout = local_fit(obj, query_points, robust_weights)
% local_fit performs the local regression at the give query
%   points
%
%   query_values = local_fit(Loess, query_points, robust_weights)
%
%
            [knn_indices,w]=obj.knn_search(query_points); % perform knn_search
            w=bsxfun(@rdivide,w,w(:,end)); % find distance/max_distance
            w=(robust_weights(knn_indices).*(1-w.^3).^3)'; % Compute weights
            query_values=nan(size(query_points,1),1); % initialize interpolation output

            % initialize x vector for regression (requires load of memory,
            % implement slow fallback for out of memory errors)
            if obj.order == 1
                xi=nan([size(knn_indices), obj.n_dims+1]); % first order constant+n_dims linear terms
            else
                xi=nan([size(knn_indices), obj.n_dims+1+sum(1:obj.n_dims)]); % first order terms + n_dims squares + sum(1:ndims) cross products
            end
            % linear terms
            xi(:,:,1)=ones(size(knn_indices));
            xi(:,:,1 + (1:obj.n_dims))=bsxfun(@minus,reshape(obj.valid_points(knn_indices,:),[size(knn_indices) obj.n_dims]),permute(query_points,[1 3 2]));
            % quadratic terms
            if obj.order==2
                % squares
                xi(:,:,1+obj.n_dims + (1:obj.n_dims))=xi(:,:,2:obj.n_dims+1).^2;
                % cross_products
                last_page=1+2*obj.n_dims;
                for cdim1=1:obj.n_dims-1
                    for cdim2=cdim1+1:obj.n_dims
                        last_page=last_page+1;
                        xi(:,:,last_page)=xi(:,:,1+cdim1).*xi(:,:,1+cdim2);
                    end
                end
            end
            xi=permute(xi,[2, 3, 1]);
            vi=obj.valid_values(knn_indices)';
            parfor count_qpoints=1:size(query_points,1)
%                 regr=lscov(xi(:,:,count_qpoints),vi(:,count_qpoints),w(:,count_qpoints));
                regr=bsxfun(@times,xi(:,:,count_qpoints),w(:,count_qpoints))\(vi(:,count_qpoints).*w(:,count_qpoints));
                query_values(count_qpoints)=regr(1);
            end
            varargout{1} = query_values;
        end

        function [idx, dist]=knn_search(obj,query_points)
% knn search performs k nearest neighbor search
%
%       [idx, dist]=knn_search(Loess, query_points) returns index and
%       distance for k nearest neighbors to query_points

            % If kd_tree is outdated rebuild it
            if obj.rebuild_kdtree
                obj.kdtree=KDTreeSearcher(obj.valid_points);
                obj.rebuild_kdtree=false;
            end
            % perform search
            [idx,dist]=obj.kdtree.knnsearch(query_points,'K',obj.n_in_span);
        end

        function reset_kdtree(obj,~,~)
% reset_kdtree deletes kd_tree and flags the tree out of date
            obj.rebuild_kdtree=true;
            obj.kdtree=[];
        end
    end
end
