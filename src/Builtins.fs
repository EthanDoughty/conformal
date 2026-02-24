module Builtins

// ---------------------------------------------------------------------------
// KNOWN_BUILTINS: set of recognized builtin function names.
// Mechanical port of analysis/builtins.py KNOWN_BUILTINS.
// Sorted alphabetically within groups for maintainability.
// ---------------------------------------------------------------------------

let KNOWN_BUILTINS : Set<string> =
    Set.ofList [
        "abs"; "acos"; "acosh"; "accumarray"; "addpath"; "all"; "angle"; "any"; "arrayfun"
        "asin"; "asinh"; "assert"; "atan"; "atan2"; "atanh"
        "bicgstab"; "bitand"; "bitshift"; "bitor"; "bitxor"; "blkdiag"
        "cat"; "cd"; "ceil"; "cell"; "cell2mat"; "cell2struct"; "cellfun"; "cgs"; "char"; "chol"
        "circshift"; "class"; "complex"; "cond"; "conj"; "contains"; "conv"; "cos"; "cosh"
        "cross"; "cumprod"; "cumsum"
        "datestr"; "dbstack"; "deal"; "deblank"; "dec2hex"; "deconv"; "deg2rad"; "delete"
        "det"; "diag"; "diff"; "dir"; "disp"; "display"; "double"
        "eig"; "eigs"; "error"; "eval"; "exist"; "exp"; "expm"; "eye"
        "false"; "fclose"; "feval"; "fft"; "fft2"; "fftshift"; "fieldnames"; "fileparts"
        "find"; "flipud"; "fliplr"; "floor"; "fopen"; "fprintf"; "fread"; "fscanf"
        "full"; "fullfile"; "fgets"; "fseek"; "ftell"; "fwrite"
        "gamrnd"; "getfield"; "gmres"
        "hex2dec"; "hex2num"; "histogram"; "horzcat"; "hypot"
        "ifft"; "ifft2"; "ifftshift"; "imag"; "inf"; "Inf"; "input"; "int16"; "int2str"
        "int32"; "int64"; "int8"; "interp1"; "interp2"; "intersect"; "inv"; "iscell"
        "ischar"; "isempty"; "isfield"; "isfinite"; "isfloat"; "isinf"; "isinteger"
        "islogical"; "ismember"; "isnan"; "isnumeric"; "isreal"; "isscalar"; "issorted"
        "issparse"; "isstring"; "isstruct"; "issymmetric"; "isvector"
        // Control System Toolbox -- Tier 1 (shape handlers in EvalBuiltins.fs)
        "lqr"; "dlqr"; "place"; "acker"; "care"; "dare"; "lyap"; "dlyap"; "obsv"; "ctrb"
        // Control System Toolbox -- Tier 2 (recognized-only)
        "ss"; "tf"; "zpk"
        "c2d"; "d2c"
        "series"; "parallel"; "feedback"
        "minreal"; "balreal"; "modred"
        "step"; "impulse"
        "bode"; "nyquist"; "margin"; "bandwidth"
        "pole"; "zero"; "dcgain"
        "kalman"; "kalmd"; "lqe"
        // Basic missing builtin
        "dot"
        "kron"
        "length"; "linspace"; "load"; "log"; "log10"; "log2"; "logical"; "logm"; "logspace"
        "lower"; "lsqnonneg"; "lu"
        "mat2cell"; "mat2str"; "max"; "mean"; "median"; "min"; "mink"; "mkdir"; "mod"
        "mvnrnd"
        "nan"; "NaN"; "nanmax"; "nanmean"; "nanmin"; "nanstd"; "nansum"; "nargin"
        "nargout"; "ndgrid"; "ndims"; "nnz"; "norm"; "normpdf"; "not"; "null"; "num2cell"
        "num2hex"; "num2str"; "numel"
        "ones"; "orderfields"; "orth"
        "pcg"; "permute"; "pinv"; "plot"; "plot3"; "poly"; "polyfit"; "polyval"; "power"
        "ppval"; "print"; "prod"
        "qr"
        "rad2deg"; "rand"; "randi"; "randn"; "rank"; "rcond"; "real"; "regexp"; "regexpi"
        "rem"; "repmat"; "reshape"; "rmpath"; "roots"; "round"
        "save"; "saveas"; "setdiff"; "setfield"; "setxor"; "sgolayfilt"; "shiftdim"; "squeeze"
        "sign"; "sin"; "single"; "sinh"; "size"; "sort"; "sparse"; "spline"; "sprank"
        "sprintf"; "sqrt"; "sqrtm"; "std"; "str2double"; "strcmp"; "strcmpi"; "strfind"
        "strjoin"; "strmatch"; "strrep"; "strsplit"; "strtrim"; "string"; "struct"
        "struct2cell"; "structfun"; "sub2ind"; "sum"; "svd"; "svds"
        "tan"; "tanh"; "textscan"; "trace"; "transpose"; "tril"; "triu"; "true"; "typecast"
        "uint16"; "uint32"; "uint64"; "uint8"; "union"; "unique"; "unwrap"; "upper"
        "var"; "vertcat"
        "warning"; "whos"; "wishrnd"
        "xor"
        "zeros"
        // Signal Processing Toolbox -- Tier 1 (shape handlers in EvalBuiltins.fs)
        "filter"; "filtfilt"
        "hamming"; "hann"; "blackman"; "kaiser"; "rectwin"; "bartlett"
        "butter"; "cheby1"; "cheby2"; "ellip"; "besself"
        "xcorr"
        // Signal Processing Toolbox -- Tier 2 (recognized-only)
        "freqz"; "pwelch"; "spectrogram"; "periodogram"; "tfestimate"; "mscohere"; "cpsd"
        "decimate"; "resample"; "downsample"; "upsample"; "medfilt1"
        "bandpass"; "lowpass"; "highpass"; "bandstop"; "findpeaks"
        "impz"; "zplane"; "grpdelay"
        // Aerospace Toolbox -- Tier 1 (shape handlers in EvalBuiltins.fs)
        "angle2dcm"; "eul2rotm"
        "quat2dcm"; "quat2rotm"
        "dcm2quat"; "rotm2quat"
        "quatmultiply"
        "quatconj"; "quatinv"; "quatnormalize"
        "quatnorm"
        "dcm2angle"
        // Aerospace Toolbox -- Tier 2 (recognized-only)
        "lla2ecef"; "ecef2lla"; "ecef2ned"; "ned2ecef"
        "atmoscoesa"; "atmosisa"
        "geocradius"; "gravitywgs84"; "geoidegm96"
        "convlength"; "convmass"; "convvel"; "convforce"; "convpres"; "convtemp"; "convangvel"
        "juliandate"; "mjuliandate"; "decyear"; "leapyear"
        "earthNutation"
        // Graphics/plotting -- recognized but no shape handler (I/O side effects only)
        "autocorr"
        "axis"; "bar"; "box"; "cla"; "clabel"; "clf"; "close"; "colorbar"; "colormap"
        "contour"; "contourf"; "drawnow"; "errorbar"; "ezcontour"; "figure"; "fill"; "gca"; "gcf"
        "grid"; "hold"; "image"; "imagesc"; "legend"; "light"; "line"; "loglog"
        "maxk"; "mesh"; "meshgrid"; "mnrnd"; "mvnpdf"; "patch"; "pause"; "pcolor"; "quiver"; "scatter"
        "semilogx"; "semilogy"; "set"; "get"; "shading"; "stem"; "subplot"; "surf"
        "surface"; "text"; "title"; "view"; "xlabel"; "xlim"; "ylabel"; "ylim"
        "zlabel"; "zlim"
    ]

// ---------------------------------------------------------------------------
// SUPPRESSED_CMD_STMTS: command-syntax statements that are silently ignored.
// Mechanical port of analysis/eval_builtins.py _SUPPRESSED_CMD_STMTS.
// ---------------------------------------------------------------------------

let SUPPRESSED_CMD_STMTS : Set<string> =
    Set.ofList [
        "addpath"; "rmpath"; "cd"; "mkdir"; "save"; "load"; "disp"; "display"
        "fprintf"; "printf"; "warning"; "error"; "assert"
        "figure"; "clf"; "cla"; "close"; "hold"; "grid"; "axis"; "xlabel"; "ylabel"
        "zlabel"; "title"; "legend"; "colorbar"; "colormap"; "shading"; "view"
        "subplot"; "plot"; "plot3"; "scatter"; "bar"; "stem"; "errorbar"; "fill"
        "mesh"; "surf"; "surface"; "contour"; "contourf"; "pcolor"; "image"; "imagesc"
        "semilogx"; "semilogy"; "loglog"; "quiver"; "patch"; "text"
        "drawnow"; "pause"; "set"; "get"; "xlim"; "ylim"; "zlim"
        "box"; "light"; "line"; "clabel"; "ezcontour"; "saveas"; "print"
        "dbstack"; "eval"; "feval"; "input"
        "delete"; "dir"; "whos"
    ]
