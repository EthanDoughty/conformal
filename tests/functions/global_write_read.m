% Test: Script calls setGlobal() then readGlobal(); reader sees shape set by writer.
% The void call to setGlobal at script level emits W_PROCEDURE_IN_EXPR (expected).
% EXPECT: warnings = 1
% EXPECT: result = matrix[5 x 5]

function setGlobal()
    global gmat;
    gmat = zeros(5, 5);
end

function y = readGlobal()
    global gmat;
    y = gmat;
end

setGlobal();  % EXPECT_WARNING: W_PROCEDURE_IN_EXPR
result = readGlobal();
