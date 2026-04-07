function conformal_check(filepath)
%CONFORMAL_CHECK Run Conformal shape analysis on a .m file.
%   conformal_check('myfile.m') analyzes the file and prints results.
%   conformal_check (no args) analyzes the current editor file.
%
%   Requires the Conformal CLI binary on your PATH.
%   Get it at: https://github.com/EthanDoughty/conformal

    if nargin < 1
        filepath = matlab.desktop.editor.getActive().Filename;
    end

    [status, output] = system(['conformal-parse "' filepath '"']);
    disp(output);
end
