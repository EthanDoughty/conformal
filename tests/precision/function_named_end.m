% Test: function named 'end' for operator overloading
% MATLAB allows end as a function name (e.g. @classname/end.m).

function e = end(f, k, n)
  dim = size(f);
  if k > length(dim)
    e = 1;
  else
    e = dim(k);
  end
end
% EXPECT: e: unknown
