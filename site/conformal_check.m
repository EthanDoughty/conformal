function conformal_check(filepath)
%CONFORMAL_CHECK Run Conformal shape analysis on a .m file.
%   conformal_check('myfile.m') analyzes the file and prints results.
%   conformal_check (no args) analyzes the current editor file.
%
%   Setup:
%     1. Download conformal-parse from:
%        https://github.com/EthanDoughty/conformal/releases
%     2. Put it somewhere on your system PATH
%     3. Put this file on your MATLAB path
%
%   Then just call:
%     >> conformal_check            % analyzes current editor file
%     >> conformal_check('ekf.m')   % analyzes a specific file

    if nargin < 1
        try
            filepath = matlab.desktop.editor.getActive().Filename;
        catch
            error('No file specified and no editor file is open.');
        end
    end

    if ~isfile(filepath)
        error('File not found: %s', filepath);
    end

    [status, output] = system(sprintf('conformal-parse "%s"', filepath));

    if status == -1
        error(['Could not find conformal-parse on PATH. ' ...
               'Download it from https://github.com/EthanDoughty/conformal/releases']);
    end

    disp(output);
end
