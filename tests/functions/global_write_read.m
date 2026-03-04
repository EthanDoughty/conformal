% Test: Script calls setGlobal() then readGlobal(); reader sees shape set by writer.
% Calling a procedure as a statement is valid MATLAB (no warning).
% EXPECT: warnings = 0
% EXPECT: result = matrix[5 x 5]

function setGlobal()
    global gmat;
    gmat = zeros(5, 5);
end

function y = readGlobal()
    global gmat;
    y = gmat;
end

setGlobal();
result = readGlobal();
