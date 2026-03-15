% SKIP_TEST
% Control flow
x = 0;
for i = 1:5
    x = x + i;
end
disp(x);

y = 1;
while y < 10
    y = y * 2;
end
disp(y);

if x > 10
    disp(1);
else
    disp(0);
end
