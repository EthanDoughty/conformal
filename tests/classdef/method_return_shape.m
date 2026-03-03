% Test: classdef method that returns a matrix - verify shape propagation
% EXPECT: warnings = 0
% EXPECT: mat = matrix[3 x 4]

classdef Factory
  properties
    rows
    cols
  end
  methods
    function obj = Factory(r, c)
      obj.rows = r;
      obj.cols = c;
    end
    function M = make_matrix(obj, r, c)
      M = zeros(r, c);
    end
  end
end

f = Factory(3.0, 4.0);
mat = f.make_matrix(3, 4);
