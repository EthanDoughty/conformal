% Test: Expanded builtin coverage (v1.11.0)
% EXPECT: warnings = 0

% === Hyperbolic trig (passthrough) ===
A = zeros(3, 4);
v = zeros(5, 1);
th1 = tanh(A);        % EXPECT: th1 = matrix[3 x 4]
ch1 = cosh(v);        % EXPECT: ch1 = matrix[5 x 1]
sh1 = sinh(A);        % EXPECT: sh1 = matrix[3 x 4]
ath1 = atanh(A);      % EXPECT: ath1 = matrix[3 x 4]
ach1 = acosh(v);      % EXPECT: ach1 = matrix[5 x 1]
ash1 = asinh(A);      % EXPECT: ash1 = matrix[3 x 4]

% === Complex / logical (passthrough) ===
cj1 = conj(A);        % EXPECT: cj1 = matrix[3 x 4]
nt1 = not(A);         % EXPECT: nt1 = matrix[3 x 4]

% === Flip/triangular (passthrough) ===
fu1 = flipud(A);      % EXPECT: fu1 = matrix[3 x 4]
fl1 = fliplr(A);      % EXPECT: fl1 = matrix[3 x 4]
tu1 = triu(A);        % EXPECT: tu1 = matrix[3 x 4]
tl1 = tril(A);        % EXPECT: tl1 = matrix[3 x 4]

% === Sort/unique (passthrough) ===
so1 = sort(v);        % EXPECT: so1 = matrix[5 x 1]
uq1 = unique(v);      % EXPECT: uq1 = matrix[5 x 1]

% === Type casts (shape preserved) ===
d1 = double(A);       % EXPECT: d1 = matrix[3 x 4]
sg1 = single(v);      % EXPECT: sg1 = matrix[5 x 1]
i8 = int8(A);         % EXPECT: i8 = matrix[3 x 4]
i16 = int16(A);       % EXPECT: i16 = matrix[3 x 4]
i32 = int32(v);       % EXPECT: i32 = matrix[5 x 1]
i64 = int64(A);       % EXPECT: i64 = matrix[3 x 4]
u8 = uint8(A);        % EXPECT: u8 = matrix[3 x 4]
u16 = uint16(v);      % EXPECT: u16 = matrix[5 x 1]
u32 = uint32(A);      % EXPECT: u32 = matrix[3 x 4]
u64 = uint64(v);      % EXPECT: u64 = matrix[5 x 1]
lg1 = logical(A);     % EXPECT: lg1 = matrix[3 x 4]
cx1 = complex(A);     % EXPECT: cx1 = matrix[3 x 4]

% === Scalar predicates ===
istr = isstruct(A);   % EXPECT: istr = scalar
irl = isreal(A);      % EXPECT: irl = scalar
isp = issparse(A);    % EXPECT: isp = scalar
ivec = isvector(v);   % EXPECT: ivec = scalar
iint = isinteger(A);  % EXPECT: iint = scalar
iflt = isfloat(A);    % EXPECT: iflt = scalar
istrg = isstring(A);  % EXPECT: istrg = scalar
isrt = issorted(v);   % EXPECT: isrt = scalar

% === Scalar queries ===
tr1 = trace(A);       % EXPECT: tr1 = scalar
rk1 = rank(A);        % EXPECT: rk1 = scalar
cn1 = cond(A);        % EXPECT: cn1 = scalar
rc1 = rcond(A);       % EXPECT: rc1 = scalar
nz1 = nnz(A);         % EXPECT: nz1 = scalar
sr1 = sprank(A);      % EXPECT: sr1 = scalar

% === Reductions (1-arg and 2-arg) ===
md1 = median(A);      % EXPECT: md1 = matrix[1 x 4]
vr1 = var(v);         % EXPECT: vr1 = matrix[1 x 1]
sd1 = std(A);         % EXPECT: sd1 = matrix[1 x 4]
md2 = median(A, 2);   % EXPECT: md2 = matrix[3 x 1]

% === Elementwise 2-arg ===
B = zeros(3, 4);
pw1 = power(A, B);    % EXPECT: pw1 = matrix[3 x 4]
hp1 = hypot(A, B);    % EXPECT: hp1 = matrix[3 x 4]
xr1 = xor(A, B);      % EXPECT: xr1 = matrix[3 x 4]
pw2 = power(A, 2);    % EXPECT: pw2 = matrix[3 x 4]

% === String returns ===
s1 = num2str(42);     % EXPECT: s1 = string
s2 = int2str(42);     % EXPECT: s2 = string
s3 = mat2str(A);      % EXPECT: s3 = string
s4 = char(65);        % EXPECT: s4 = string
s5 = string(42);      % EXPECT: s5 = string
s6 = sprintf('x=%d', 5);  % EXPECT: s6 = string

% === randi ===
r0 = randi(10);       % EXPECT: r0 = scalar
r1 = randi(10, 3);    % EXPECT: r1 = matrix[3 x 3]
r2 = randi(10, 3, 4); % EXPECT: r2 = matrix[3 x 4]

% === find ===
f1 = find(A);         % EXPECT: f1 = matrix[1 x None]

% === cat ===
C = zeros(2, 4);
ct1 = cat(1, A, C);   % EXPECT: ct1 = matrix[5 x 4]
D = zeros(3, 2);
ct2 = cat(2, A, D);   % EXPECT: ct2 = matrix[3 x 6]

% === I/O (no warning, result is unknown) ===
disp(A);
fprintf('hello\n');
error('msg');
warning('msg');
