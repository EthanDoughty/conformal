% Test: interp1 follows the query points, not the table values
% EXPECT: warnings = 0

alt = zeros(1, 6);
temp = zeros(1, 6);
hq = zeros(1, 51);

% 3-arg, scalar query -> scalar
Th = interp1(alt, temp, 2500);
% EXPECT: Th = scalar

% 3-arg, vector query -> query's shape, not the table's
Tq = interp1(alt, temp, hq);
% EXPECT: Tq = matrix[1 x 51]

% 4-arg with a trailing method string -> still the query's shape
Tq4 = interp1(alt, temp, hq, 'linear');
% EXPECT: Tq4 = matrix[1 x 51]

% 5-arg with two trailing method/extrap strings -> still the query's shape
Tq5 = interp1(alt, temp, hq, 'linear', 'extrap');
% EXPECT: Tq5 = matrix[1 x 51]

% 3-arg METHOD form: interp1(v, xq, 'linear') -- args[2] is a method name,
% NOT a query. The case a naive "return args[2]" fix would break.
Tm = interp1(temp, hq, 'linear');
% EXPECT: Tm = matrix[1 x 51]

% 2-arg form: interp1(v, xq) -> xq's shape
T2 = interp1(temp, hq);
% EXPECT: T2 = matrix[1 x 51]
