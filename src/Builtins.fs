module Builtins

// ---------------------------------------------------------------------------
// KNOWN_BUILTINS: set of recognized builtin function names.
// Mechanical port of analysis/builtins.py KNOWN_BUILTINS.
// Sorted alphabetically within groups for maintainability.
// ---------------------------------------------------------------------------

let KNOWN_BUILTINS : Set<string> =
    Set.ofList [
        "abs"; "acos"; "acosd"; "acosh"; "accumarray"; "addpath"; "all"; "almanac"; "angle"; "any"; "arrayfun"
        "asin"; "asinh"; "asind"; "assert"; "atan"; "atan2"; "atan2d"; "atand"; "atanh"
        "bicgstab"; "bitand"; "bitshift"; "bitor"; "bitxor"; "blkdiag"; "boxcar"; "bsxfun"
        "cat"; "cd"; "ceil"; "cell"; "cell2mat"; "cell2struct"; "cellfun"; "cgs"; "char"; "chi2inv"; "chol"
        "circshift"; "class"; "complex"; "cond"; "conj"; "contains"; "conv"; "copyfile"; "cos"; "cosd"; "cosh"
        "cotd"; "createOptimProblem"; "cross"; "cscd"; "cumprod"; "cumsum"
        "datestr"; "dbstack"; "deal"; "deblank"; "dec2hex"; "deconv"; "deg2rad"; "delete"
        "det"; "diag"; "diff"; "dir"; "disp"; "display"; "double"
        "eig"; "eigs"; "error"; "eval"; "exist"; "exp"; "expm"; "eye"
        "fitgmdist"; "fmincon"; "fminunc"; "fminsearch"; "fsolve"
        "false"; "fclose"; "feval"; "fft"; "fft2"; "fftshift"; "fieldnames"; "fileparts"
        "find"; "flipud"; "fliplr"; "floor"; "fopen"; "fprintf"; "fread"; "fscanf"
        "full"; "fullfile"; "fgets"; "fseek"; "ftell"; "fwrite"
        "gamrnd"; "geodetic2ecef"; "geotiffinfo"; "geotiffread"; "geotiffwrite"; "getfield"; "gmres"; "gmdistribution"
        "hanning"; "hex2dec"; "hex2num"; "hist"; "histc"; "histogram"; "horzcat"; "hypot"
        "ifft"; "ifft2"; "ifftshift"; "imag"; "inf"; "Inf"; "input"; "int16"; "int2str"
        "interpm"; "interpft"
        "int32"; "int64"; "int8"; "interp1"; "interp2"; "intersect"; "inv"; "iscell"
        "kmeans"
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
        "lsqcurvefit"; "lsqnonlin"
        "lower"; "lsqnonneg"; "lu"
        "mat2cell"; "mat2str"; "max"; "mean"; "median"; "min"; "mink"; "mkdir"; "mod"; "movefile"
        "mvnrnd"
        "nan"; "NaN"; "nanmax"; "nanmean"; "nanmin"; "nanstd"; "nansum"; "nanvar"
        "nchoosek"; "ncinfo"; "ncread"; "ncreadatt"; "ncwrite"; "normcdf"; "norminv"
        "ndgrid"; "ndims"; "nnz"; "norm"; "normpdf"; "not"; "null"; "num2cell"
        "num2hex"; "num2str"; "numel"
        "ones"; "optimget"; "optimoptions"; "optimset"; "orderfields"; "orth"
        "patternsearch"; "pcg"; "pdist"; "pdist2"; "permute"; "pinv"; "plot"; "plot3"
        "poly"; "polyfit"; "polyval"; "power"; "projfwd"; "projinv"; "psoptimset"
        "ppval"; "print"; "prod"
        "qr"
        "rad2deg"; "rand"; "randi"; "randn"; "rank"; "rcond"; "real"; "regexp"; "regexpi"; "regexprep"
        "rem"; "repmat"; "reshape"; "rmfield"; "rmpath"; "roots"; "round"
        "run"
        "save"; "saveas"; "setdiff"; "setfield"; "setxor"; "sgolayfilt"; "shiftdim"; "squeeze"
        "secd"; "sign"; "sin"; "sind"; "single"; "sinh"; "size"; "sort"; "sparse"; "spline"; "sprank"
        "shaperead"
        "sprintf"; "sqrt"; "sqrtm"; "std"; "str2double"; "strcmp"; "strcmpi"; "strfind"
        "strjoin"; "strmatch"; "strrep"; "strsplit"; "strtrim"; "string"; "struct"
        "struct2cell"; "structfun"; "sub2ind"; "sum"; "svd"; "svds"
        "tan"; "tand"; "tanh"; "textscan"; "tinv"; "trace"; "transpose"; "tril"; "triu"; "true"; "ttest"; "ttest2"; "tukeywin"; "typecast"
        "uint16"; "uint32"; "uint64"; "uint8"; "union"; "unique"; "unwrap"; "upper"
        "var"; "vertcat"
        "warning"; "whos"; "wishrnd"; "wrapTo180"; "wrapTo360"; "wrapToPi"
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
        // Batch 6a passthrough additions (need to be recognized)
        "sortrows"; "flip"; "gradient"; "detrend"
        "movmean"; "movstd"; "movmedian"; "movsum"; "movmax"; "movmin"
        "normalize"; "rescale"; "cummax"; "cummin"; "zscore"
        "imfilter"; "imgaussfilt"; "medfilt2"; "imadjust"; "histeq"
        "im2double"; "im2uint8"; "im2single"; "mat2gray"; "imbinarize"
        "imerode"; "imdilate"; "imopen"; "imclose"; "imfill"; "imcomplement"
        "adapthisteq"; "wiener2"; "edge"; "bwlabel"
        "simplify"; "expand"; "subs"
        "undistortImage"
        "rgb2gray"
        // Batch 6b: Constant-dimension robotics/CV transforms
        "axang2rotm"; "tform2rotm"; "estimateFundamentalMatrix"; "estimateEssentialMatrix"
        "rotm2tform"; "trvec2tform"; "eul2tform"; "axang2tform"; "quat2tform"
        "rotm2eul"; "tform2trvec"; "tform2eul"; "quat2eul"
        "rotm2axang"; "tform2axang"; "eul2quat"; "axang2quat"; "quat2axang"
        // Batch 6c: Statistics reductions (mode/kurtosis/skewness/range) and random generators
        "mode"; "kurtosis"; "skewness"; "range"
        "normrnd"; "exprnd"; "unifrnd"; "poissrnd"; "chi2rnd"; "binornd"; "betarnd"
        // Batch 6d: Complex handlers
        "cov"; "corrcoef"; "rot90"; "conv2"; "num2cell"; "jacobian"
        // Batch 6e: Multi-return handlers
        "pca"; "ind2sub"; "linprog"; "quadprog"
        // Batch 6f: Recognized-only (no shape rules)
        // Image Processing
        "imread"; "imresize"; "imrotate"; "imcrop"; "imshow"; "imwrite"
        "imwarp"; "regionprops"; "bwconncomp"; "bwareaopen"; "bwareafilt"; "padarray"
        // Statistics
        "fitlm"; "fitcsvm"; "fitctree"; "predict"; "crossval"; "fitcecoc"
        "squareform"; "linkage"; "prctile"; "quantile"
        // Optimization
        "ga"; "particleswarm"; "intlinprog"; "fgoalattain"
        // Deep Learning
        "dlarray"; "trainNetwork"; "trainingOptions"; "classify"
        "sigmoid"; "relu"; "softmax"
        // Symbolic
        "sym"; "matlabFunction"; "vpa"; "hessian"
        // Computer Vision
        "detectSURFFeatures"; "extractFeatures"; "matchFeatures"
        // Communications
        "qammod"; "qamdemod"; "pskmod"; "pskdemod"; "awgn"; "biterr"; "symerr"; "rcosdesign"
        // Data I/O
        "readtable"; "readmatrix"; "writetable"; "writematrix"
        "table"; "array2table"; "table2array"; "height"; "width"
        // Misc
        "groupsummary"; "splitapply"; "varfun"; "rowfun"; "vecnorm"
        // Graphics/plotting -- recognized but no shape handler (I/O side effects only)
        "annotation"; "autocorr"; "axes"
        "axis"; "bar"; "box"; "bufferm"; "caxis"; "cla"; "clabel"; "clf"; "close"; "colorbar"; "colormap"
        "contour"; "contourf"; "drawnow"; "errorbar"; "ezcontour"; "figure"; "fill"; "gca"; "gcf"
        "linkaxes"
        "grid"; "hold"; "image"; "imagesc"; "legend"; "light"; "line"; "loglog"
        "maxk"; "mesh"; "meshgrid"; "mnrnd"; "mvnpdf"; "patch"; "pause"; "pcolor"; "quiver"; "scatter"
        "semilogx"; "semilogy"; "set"; "get"; "shading"; "stem"; "subplot"; "surf"
        "surface"; "text"; "title"; "uicontrol"; "uimenu"; "view"; "xlabel"; "xlim"; "ylabel"; "ylim"
        "zlabel"; "zlim"
    ]

// ---------------------------------------------------------------------------
// CODER_UNSUPPORTED_BUILTINS: builtins with no MATLAB Coder equivalent.
// Conservative set: high-confidence entries only.
// Note: MathWorks expands Coder support over time; revisit periodically.
// ---------------------------------------------------------------------------

let CODER_UNSUPPORTED_BUILTINS : Set<string> =
    Set.ofList [
        // Dynamic evaluation
        "eval"; "evalin"; "feval"; "assignin"; "evalc"
        // Interactive / debugging
        "input"; "keyboard"; "dbstop"; "dbclear"; "dbcont"
        // Graphics / plotting
        "figure"; "plot"; "plot3"; "subplot"; "surf"; "mesh"; "imagesc"; "scatter"
        "bar"; "histogram"; "stem"; "stairs"; "contour"; "pie"; "area"; "pareto"; "errorbar"
        // UI
        "uicontrol"; "uimenu"; "uipanel"; "uitable"; "msgbox"; "inputdlg"; "questdlg"
        // Introspection / help
        "diary"; "type"; "doc"; "help"; "lookfor"; "which"; "what"; "who"; "whos"
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
        "axes"; "caxis"; "linkaxes"; "annotation"; "uicontrol"; "uimenu"
    ]
