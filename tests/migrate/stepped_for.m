% Test stepped for-loop translation
for i = 1:2:9
    x = i * 2;
end
step = -1;
for j = 10:step:1
    y = j;
end
