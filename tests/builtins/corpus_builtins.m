% corpus_builtins.m — verify dogfood corpus builtins don't produce W_UNKNOWN_FUNCTION
% EXPECT: warnings = 0

% Group 1: Case variants (NaN, Inf)
a = NaN(3, 4);
b = Inf(2, 2);
c = NaN;
d = Inf;

% Group 2: string-returning
p = fullfile('dir', 'subdir', 'file.txt');

% Group 3: scalar-returning
r1 = strcmpi('hello', 'HELLO');
r2 = strcmp('abc', 'abc');
r3 = exist('myfile.m', 'file');
r4 = str2double('3.14');

% Group 4: passthrough (shape preserved)
x = zeros(5, 3);
y1 = squeeze(x);
y2 = fftshift(x);
y3 = ifftshift(x);
y4 = unwrap(x);
y5 = deg2rad(x);
y6 = rad2deg(x);
y7 = angle(x);
y8 = sgolayfilt(x, 3, 7);

% Group 5: type cast
v = uint8([1 2 3]);
tc = typecast(v, 'uint16');

% Group 6: NaN-ignoring reductions
m = [1 NaN 3; 4 5 NaN];
nm = nanmean(m);
ns = nansum(m);
nstd = nanstd(m);
nmin = nanmin(m);
nmax = nanmax(m);

% Group 7: I/O builtins (no handler, recognized only)
load('data.mat');
save('out.mat', 'x');
mkdir('newdir');
assert(true);
