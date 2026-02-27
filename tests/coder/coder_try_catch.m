% Test: W_CODER_TRY_CATCH fires for try/catch blocks.
% MODE: coder

% EXPECT: warnings = 1

try
    x = 1;
catch
    x = 0;
end
