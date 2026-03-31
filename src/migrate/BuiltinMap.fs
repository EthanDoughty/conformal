module BuiltinMap

open PyAst

type ArgTransform =
    | Direct
    | TupleFirstN of int
    | AttrStyle of string
    | AttrStyleDim
    | MaxShape
    | SizeAttr
    | MinMaxDispatch of elemwise: string  // np.maximum / np.minimum for 2-arg
    | FlatNonzero                         // find(A) -> np.flatnonzero(A)
    | WithKwarg of key: string * value: float  // adds a keyword argument
    | StrjoinStyle                              // strjoin(parts, sep) -> sep.join(parts)
    | SortStyle                                 // [s, i] = sort(A) needs special multi-return
    | BinOpStyle of string                        // strcmp(a,b) -> a == b
    | IsEmptyStyle                                // isempty(A) -> A.size == 0
    | RaiseStyle                                  // error(msg) -> raise ValueError(msg)
    | BsxfunStyle                                 // bsxfun(@op, A, B) -> A op B
    | PermuteStyle                                // permute(A, order) -> np.transpose(A, order-1)
    | Sub2IndStyle                                // sub2ind(sz, i, j) -> np.ravel_multi_index((i-1, j-1), sz, order='F')
    | Ind2SubStyle                                // ind2sub(sz, ind) -> np.unravel_index(ind-1, sz, order='F')
    | SprintfStyle                                // sprintf(fmt, args) -> fmt % (args,)
    | FprintfStyle                                // fprintf(fmt, args) -> print(fmt % (args,))
    | MethodStyle of string                          // lower(s) -> s.lower()
    | StrcmpiStyle                                   // strcmpi(a,b) -> a.lower() == b.lower()
    | CatStyle                                       // cat(dim, A, B) -> np.concatenate((A, B), axis=dim-1)
    | DimArgStyle                                    // sum(A, dim) -> np.sum(A, axis=dim-1)
    | CellfunStyle                                   // cellfun(@f, C) -> [f(x) for x in C]
    | ExistStyle                                    // exist('x','var') -> x is not None; exist(f,'file') -> os.path.exists(f)
    | ClassStyle                                    // class(x) -> type(x).__name__
    | IsTypeStyle of string                          // ischar(x) -> isinstance(x, str)
    | StructStyle                                    // struct('a',1,'b',2) -> SimpleNamespace(a=1, b=2)
    | FevalStyle                                    // feval(f, a, b) -> f(a, b)
    | RegexpStyle                                   // regexp(str, pat) -> re.search(pat, str)
    | RegexpRepStyle                                // regexprep(str, pat, rep) -> re.sub(pat, rep, str)
    | NanFullStyle                                  // nan(m, n) -> np.full((m, n), np.nan)
    | FileIoStyle of string                         // fwrite(fd, data) -> sys.stderr.write(data) for fd=2

type BuiltinMapping = {
    pythonFunc: string
    argTransform: ArgTransform
    needsOrderF: bool
}

let private builtinTable =
    dict [
        "zeros",     { pythonFunc = "np.zeros";          argTransform = TupleFirstN 2; needsOrderF = false }
        "ones",      { pythonFunc = "np.ones";           argTransform = TupleFirstN 2; needsOrderF = false }
        "eye",       { pythonFunc = "np.eye";            argTransform = Direct;        needsOrderF = false }
        "rand",      { pythonFunc = "np.random.rand";    argTransform = Direct;        needsOrderF = false }
        "randn",     { pythonFunc = "np.random.randn";   argTransform = Direct;        needsOrderF = false }
        "linspace",  { pythonFunc = "np.linspace";       argTransform = Direct;        needsOrderF = false }
        "reshape",   { pythonFunc = "np.reshape";        argTransform = TupleFirstN 2; needsOrderF = true  }
        "size",      { pythonFunc = ".shape";            argTransform = AttrStyleDim; needsOrderF = false }
        "length",    { pythonFunc = "max(.shape)";       argTransform = MaxShape;      needsOrderF = false }
        "numel",     { pythonFunc = ".size";             argTransform = SizeAttr;      needsOrderF = false }
        "inv",       { pythonFunc = "np.linalg.inv";     argTransform = Direct;        needsOrderF = false }
        "det",       { pythonFunc = "np.linalg.det";     argTransform = Direct;        needsOrderF = false }
        "eig",       { pythonFunc = "np.linalg.eig";     argTransform = Direct;        needsOrderF = false }
        "svd",       { pythonFunc = "np.linalg.svd";     argTransform = Direct;        needsOrderF = false }
        "pinv",      { pythonFunc = "np.linalg.pinv";    argTransform = Direct;        needsOrderF = false }
        "norm",      { pythonFunc = "np.linalg.norm";    argTransform = Direct;        needsOrderF = false }
        "diag",      { pythonFunc = "np.diag";           argTransform = Direct;        needsOrderF = false }
        "sum",       { pythonFunc = "np.sum";            argTransform = DimArgStyle;   needsOrderF = false }
        "max",       { pythonFunc = "np.max";            argTransform = MinMaxDispatch "np.maximum"; needsOrderF = false }
        "min",       { pythonFunc = "np.min";            argTransform = MinMaxDispatch "np.minimum"; needsOrderF = false }
        "abs",       { pythonFunc = "np.abs";            argTransform = Direct;        needsOrderF = false }
        "sqrt",      { pythonFunc = "np.sqrt";           argTransform = Direct;        needsOrderF = false }
        "exp",       { pythonFunc = "np.exp";            argTransform = Direct;        needsOrderF = false }
        "log",       { pythonFunc = "np.log";            argTransform = Direct;        needsOrderF = false }
        "disp",      { pythonFunc = "print";             argTransform = Direct;        needsOrderF = false }
        "fprintf",   { pythonFunc = "print";             argTransform = FprintfStyle;   needsOrderF = false }
        "sprintf",   { pythonFunc = "";                  argTransform = SprintfStyle;   needsOrderF = false }
        "sort",      { pythonFunc = "np.sort";           argTransform = SortStyle;     needsOrderF = false }
        "find",      { pythonFunc = "np.flatnonzero";    argTransform = FlatNonzero;   needsOrderF = false }
        "transpose", { pythonFunc = ".T";                argTransform = AttrStyle "T"; needsOrderF = false }
        "true",      { pythonFunc = "True";              argTransform = Direct;        needsOrderF = false }
        "false",     { pythonFunc = "False";             argTransform = Direct;        needsOrderF = false }
        // Trig functions
        "sin",       { pythonFunc = "np.sin";            argTransform = Direct;        needsOrderF = false }
        "cos",       { pythonFunc = "np.cos";            argTransform = Direct;        needsOrderF = false }
        "tan",       { pythonFunc = "np.tan";            argTransform = Direct;        needsOrderF = false }
        "asin",      { pythonFunc = "np.arcsin";         argTransform = Direct;        needsOrderF = false }
        "acos",      { pythonFunc = "np.arccos";         argTransform = Direct;        needsOrderF = false }
        "atan",      { pythonFunc = "np.arctan";         argTransform = Direct;        needsOrderF = false }
        "atan2",     { pythonFunc = "np.arctan2";        argTransform = Direct;        needsOrderF = false }
        "sinh",      { pythonFunc = "np.sinh";           argTransform = Direct;        needsOrderF = false }
        "cosh",      { pythonFunc = "np.cosh";           argTransform = Direct;        needsOrderF = false }
        "tanh",      { pythonFunc = "np.tanh";           argTransform = Direct;        needsOrderF = false }
        // Math functions
        "floor",     { pythonFunc = "np.floor";          argTransform = Direct;        needsOrderF = false }
        "ceil",      { pythonFunc = "np.ceil";           argTransform = Direct;        needsOrderF = false }
        "round",     { pythonFunc = "np.round";          argTransform = Direct;        needsOrderF = false }
        "mod",       { pythonFunc = "np.mod";            argTransform = Direct;        needsOrderF = false }
        "rem",       { pythonFunc = "np.remainder";      argTransform = Direct;        needsOrderF = false }
        "sign",      { pythonFunc = "np.sign";           argTransform = Direct;        needsOrderF = false }
        "log2",      { pythonFunc = "np.log2";           argTransform = Direct;        needsOrderF = false }
        "log10",     { pythonFunc = "np.log10";          argTransform = Direct;        needsOrderF = false }
        "real",      { pythonFunc = "np.real";           argTransform = Direct;        needsOrderF = false }
        "imag",      { pythonFunc = "np.imag";           argTransform = Direct;        needsOrderF = false }
        "conj",      { pythonFunc = "np.conj";           argTransform = Direct;        needsOrderF = false }
        // Statistics
        "mean",      { pythonFunc = "np.mean";           argTransform = DimArgStyle;   needsOrderF = false }
        "std",       { pythonFunc = "np.std";            argTransform = WithKwarg ("ddof", 1.0); needsOrderF = false }
        "var",       { pythonFunc = "np.var";            argTransform = WithKwarg ("ddof", 1.0); needsOrderF = false }
        "median",    { pythonFunc = "np.median";         argTransform = DimArgStyle;   needsOrderF = false }
        "prod",      { pythonFunc = "np.prod";           argTransform = DimArgStyle;   needsOrderF = false }
        "cumsum",    { pythonFunc = "np.cumsum";         argTransform = DimArgStyle;   needsOrderF = false }
        "cumprod",   { pythonFunc = "np.cumprod";        argTransform = DimArgStyle;   needsOrderF = false }
        // Logical
        "any",       { pythonFunc = "np.any";            argTransform = DimArgStyle;   needsOrderF = false }
        "all",       { pythonFunc = "np.all";            argTransform = DimArgStyle;   needsOrderF = false }
        "isnan",     { pythonFunc = "np.isnan";          argTransform = Direct;        needsOrderF = false }
        "isinf",     { pythonFunc = "np.isinf";          argTransform = Direct;        needsOrderF = false }
        "isempty",   { pythonFunc = "";                  argTransform = IsEmptyStyle;   needsOrderF = false }
        // Array manipulation
        "repmat",    { pythonFunc = "np.tile";           argTransform = Direct;        needsOrderF = false }
        "fliplr",    { pythonFunc = "np.fliplr";         argTransform = Direct;        needsOrderF = false }
        "flipud",    { pythonFunc = "np.flipud";         argTransform = Direct;        needsOrderF = false }
        "kron",      { pythonFunc = "np.kron";           argTransform = Direct;        needsOrderF = false }
        "cross",     { pythonFunc = "np.cross";          argTransform = Direct;        needsOrderF = false }
        "dot",       { pythonFunc = "np.dot";            argTransform = Direct;        needsOrderF = false }
        "unique",    { pythonFunc = "np.unique";         argTransform = Direct;        needsOrderF = false }
        // Type conversion
        "complex",   { pythonFunc = "complex";           argTransform = Direct;        needsOrderF = false }
        "double",    { pythonFunc = "np.float64";        argTransform = Direct;        needsOrderF = false }
        "single",    { pythonFunc = "np.float32";        argTransform = Direct;        needsOrderF = false }
        "int32",     { pythonFunc = "np.int32";          argTransform = Direct;        needsOrderF = false }
        "int64",     { pythonFunc = "np.int64";          argTransform = Direct;        needsOrderF = false }
        "uint8",     { pythonFunc = "np.uint8";          argTransform = Direct;        needsOrderF = false }
        "logical",   { pythonFunc = "bool";              argTransform = Direct;        needsOrderF = false }
        // String
        "num2str",   { pythonFunc = "str";               argTransform = Direct;        needsOrderF = false }
        "str2double",{ pythonFunc = "float";             argTransform = Direct;        needsOrderF = false }
        "strcmp",     { pythonFunc = "==";                argTransform = BinOpStyle "=="; needsOrderF = false }
        "lower",     { pythonFunc = "lower";             argTransform = MethodStyle "lower"; needsOrderF = false }
        "upper",     { pythonFunc = "upper";             argTransform = MethodStyle "upper"; needsOrderF = false }
        // Diagnostics
        "warning",   { pythonFunc = "warnings.warn";     argTransform = Direct;         needsOrderF = false }
        // Broadcasting
        "bsxfun",    { pythonFunc = "";                  argTransform = BsxfunStyle;    needsOrderF = false }
        // Dimension reordering
        "permute",   { pythonFunc = "np.transpose";      argTransform = PermuteStyle;   needsOrderF = false }
        "ipermute",  { pythonFunc = "np.transpose";      argTransform = PermuteStyle;   needsOrderF = false }
        // Index conversion
        "sub2ind",   { pythonFunc = "np.ravel_multi_index"; argTransform = Sub2IndStyle; needsOrderF = false }
        "ind2sub",   { pythonFunc = "np.unravel_index";    argTransform = Ind2SubStyle;  needsOrderF = false }
        // Other
        "error",     { pythonFunc = "ValueError";        argTransform = RaiseStyle;     needsOrderF = false }
        "assert",    { pythonFunc = "assert";            argTransform = Direct;         needsOrderF = false }
        // Array manipulation (additional)
        "diff",      { pythonFunc = "np.diff";           argTransform = DimArgStyle;    needsOrderF = false }
        "cat",       { pythonFunc = "np.concatenate";    argTransform = CatStyle;       needsOrderF = false }
        "horzcat",   { pythonFunc = "np.hstack";         argTransform = Direct;         needsOrderF = false }
        "vertcat",   { pythonFunc = "np.vstack";         argTransform = Direct;         needsOrderF = false }
        "repelem",   { pythonFunc = "np.repeat";         argTransform = Direct;         needsOrderF = false }
        "ismember",  { pythonFunc = "np.isin";           argTransform = Direct;         needsOrderF = false }
        // Type conversion (additional)
        "uint16",    { pythonFunc = "np.uint16";         argTransform = Direct;         needsOrderF = false }
        "uint32",    { pythonFunc = "np.uint32";         argTransform = Direct;         needsOrderF = false }
        "uint64",    { pythonFunc = "np.uint64";         argTransform = Direct;         needsOrderF = false }
        "int8",      { pythonFunc = "np.int8";           argTransform = Direct;         needsOrderF = false }
        "int16",     { pythonFunc = "np.int16";          argTransform = Direct;         needsOrderF = false }
        "char",      { pythonFunc = "chr";               argTransform = Direct;         needsOrderF = false }
        "cell",      { pythonFunc = "list";              argTransform = Direct;         needsOrderF = false }
        // String (additional)
        "strcmpi",   { pythonFunc = "";                  argTransform = StrcmpiStyle; needsOrderF = false }
        "strtrim",   { pythonFunc = "strip";             argTransform = MethodStyle "strip"; needsOrderF = false }
        // Struct/field queries
        "isfield",   { pythonFunc = "hasattr";           argTransform = Direct;         needsOrderF = false }
        "fieldnames", { pythonFunc = "vars";             argTransform = Direct;         needsOrderF = false }
        // File/path
        "fullfile",  { pythonFunc = "os.path.join";      argTransform = Direct;         needsOrderF = false }
        // Plotting (pass through to matplotlib)
        "figure",    { pythonFunc = "plt.figure";        argTransform = Direct;         needsOrderF = false }
        "plot",      { pythonFunc = "plt.plot";          argTransform = Direct;         needsOrderF = false }
        "subplot",   { pythonFunc = "plt.subplot";       argTransform = Direct;         needsOrderF = false }
        "title",     { pythonFunc = "plt.title";         argTransform = Direct;         needsOrderF = false }
        "xlabel",    { pythonFunc = "plt.xlabel";        argTransform = Direct;         needsOrderF = false }
        "ylabel",    { pythonFunc = "plt.ylabel";        argTransform = Direct;         needsOrderF = false }
        "legend",    { pythonFunc = "plt.legend";        argTransform = Direct;         needsOrderF = false }
        "hold",      { pythonFunc = "plt.hold";          argTransform = Direct;         needsOrderF = false }
        "grid",      { pythonFunc = "plt.grid";          argTransform = Direct;         needsOrderF = false }
        "axis",      { pythonFunc = "plt.axis";          argTransform = Direct;         needsOrderF = false }
        "close",     { pythonFunc = "plt.close";         argTransform = Direct;         needsOrderF = false }
        "mesh",      { pythonFunc = "plt.pcolormesh";   argTransform = Direct;         needsOrderF = false }
        "surf",      { pythonFunc = "plt.plot_surface";  argTransform = Direct;         needsOrderF = false }
        "contour",   { pythonFunc = "plt.contour";      argTransform = Direct;         needsOrderF = false }
        "colorbar",  { pythonFunc = "plt.colorbar";     argTransform = Direct;         needsOrderF = false }
        "scatter",   { pythonFunc = "plt.scatter";      argTransform = Direct;         needsOrderF = false }
        "bar",       { pythonFunc = "plt.bar";          argTransform = Direct;         needsOrderF = false }
        "hist",      { pythonFunc = "plt.hist";         argTransform = Direct;         needsOrderF = false }
        "imagesc",   { pythonFunc = "plt.imshow";       argTransform = Direct;         needsOrderF = false }
        // Interpolation / integration
        "interp1",   { pythonFunc = "np.interp";        argTransform = Direct;         needsOrderF = false }
        "trapz",     { pythonFunc = "np.trapz";         argTransform = Direct;         needsOrderF = false }
        // Rounding / misc
        "fix",       { pythonFunc = "np.fix";           argTransform = Direct;         needsOrderF = false }
        "sparse",    { pythonFunc = "scipy.sparse.csr_matrix"; argTransform = Direct;  needsOrderF = false }
        "accumarray", { pythonFunc = "np.add.at";       argTransform = Direct;         needsOrderF = false }
        // String additional
        "strsplit",  { pythonFunc = "split";            argTransform = MethodStyle "split"; needsOrderF = false }
        "strjoin",   { pythonFunc = "join";             argTransform = StrjoinStyle;    needsOrderF = false }
        "contains",  { pythonFunc = "in";               argTransform = BinOpStyle "in"; needsOrderF = false }
        // I/O
        "load",      { pythonFunc = "scipy.io.loadmat"; argTransform = Direct;         needsOrderF = false }
        "save",      { pythonFunc = "scipy.io.savemat"; argTransform = Direct;         needsOrderF = false }
        // Sparse (additional)
        "speye",     { pythonFunc = "scipy.sparse.eye"; argTransform = Direct;         needsOrderF = false }
        "spdiags",   { pythonFunc = "scipy.sparse.diags"; argTransform = Direct;       needsOrderF = false }
        "full",      { pythonFunc = "np.array";         argTransform = Direct;         needsOrderF = false }
        "nnz",       { pythonFunc = "np.count_nonzero"; argTransform = Direct;         needsOrderF = false }
        "spy",       { pythonFunc = "plt.spy";          argTransform = Direct;         needsOrderF = false }
        // Set operations
        "setdiff",   { pythonFunc = "np.setdiff1d";     argTransform = Direct;         needsOrderF = false }
        "intersect", { pythonFunc = "np.intersect1d";   argTransform = Direct;         needsOrderF = false }
        "union",     { pythonFunc = "np.union1d";       argTransform = Direct;         needsOrderF = false }
        // Functional
        "cellfun",   { pythonFunc = "";                 argTransform = CellfunStyle;   needsOrderF = false }
        "arrayfun",  { pythonFunc = "";                 argTransform = CellfunStyle;   needsOrderF = false }
        // Miscellaneous
        "triu",      { pythonFunc = "np.triu";          argTransform = Direct;         needsOrderF = false }
        "tril",      { pythonFunc = "np.tril";          argTransform = Direct;         needsOrderF = false }
        "trace",     { pythonFunc = "np.trace";         argTransform = Direct;         needsOrderF = false }
        "poly",      { pythonFunc = "np.poly";          argTransform = Direct;         needsOrderF = false }
        "roots",     { pythonFunc = "np.roots";         argTransform = Direct;         needsOrderF = false }
        "polyval",   { pythonFunc = "np.polyval";       argTransform = Direct;         needsOrderF = false }
        "polyfit",   { pythonFunc = "np.polyfit";       argTransform = Direct;         needsOrderF = false }
        "conv",      { pythonFunc = "np.convolve";      argTransform = Direct;         needsOrderF = false }
        "deconv",    { pythonFunc = "np.polydiv";       argTransform = Direct;         needsOrderF = false }
        // Type queries
        "class",     { pythonFunc = "type";             argTransform = ClassStyle;     needsOrderF = false }
        // Existence checks
        "exist",     { pythonFunc = "os.path.exists";   argTransform = ExistStyle;     needsOrderF = false }
        "fft",       { pythonFunc = "np.fft.fft";       argTransform = Direct;         needsOrderF = false }
        "ifft",      { pythonFunc = "np.fft.ifft";      argTransform = Direct;         needsOrderF = false }
        "fft2",      { pythonFunc = "np.fft.fft2";      argTransform = Direct;         needsOrderF = false }
        "ifft2",     { pythonFunc = "np.fft.ifft2";     argTransform = Direct;         needsOrderF = false }
        // Type queries (additional)
        "ischar",    { pythonFunc = "isinstance";       argTransform = IsTypeStyle "str"; needsOrderF = false }
        "isstring",  { pythonFunc = "isinstance";       argTransform = IsTypeStyle "str"; needsOrderF = false }
        "isnumeric", { pythonFunc = "isinstance";       argTransform = IsTypeStyle "(int, float, np.ndarray)"; needsOrderF = false }
        "islogical", { pythonFunc = "isinstance";       argTransform = IsTypeStyle "(bool, np.bool_)"; needsOrderF = false }
        "iscell",    { pythonFunc = "isinstance";       argTransform = IsTypeStyle "list"; needsOrderF = false }
        "isstruct",  { pythonFunc = "isinstance";       argTransform = IsTypeStyle "types.SimpleNamespace"; needsOrderF = false }
        "isfloat",   { pythonFunc = "isinstance";       argTransform = IsTypeStyle "(float, np.floating)"; needsOrderF = false }
        "isinteger", { pythonFunc = "isinstance";       argTransform = IsTypeStyle "(int, np.integer)"; needsOrderF = false }
        "isscalar",  { pythonFunc = "np.isscalar";      argTransform = Direct;         needsOrderF = false }
        "isvector",  { pythonFunc = "np.ndim";          argTransform = Direct;         needsOrderF = false }
        // Struct constructor
        "struct",    { pythonFunc = "types.SimpleNamespace"; argTransform = StructStyle; needsOrderF = false }
        // String (additional)
        "strncmp",   { pythonFunc = "";                 argTransform = BinOpStyle "=="; needsOrderF = false }
        "strfind",   { pythonFunc = "find";             argTransform = MethodStyle "find"; needsOrderF = false }
        "strrep",    { pythonFunc = "replace";          argTransform = MethodStyle "replace"; needsOrderF = false }
        // Degree trig (wrapped: cosd(x) -> np.cos(np.radians(x)))
        "cosd",      { pythonFunc = "np.cos";           argTransform = Direct;         needsOrderF = false }
        "sind",      { pythonFunc = "np.sin";           argTransform = Direct;         needsOrderF = false }
        "tand",      { pythonFunc = "np.tan";           argTransform = Direct;         needsOrderF = false }
        // Plotting (additional)
        "drawnow",   { pythonFunc = "plt.draw";         argTransform = Direct;         needsOrderF = false }
        "clf",       { pythonFunc = "plt.clf";          argTransform = Direct;         needsOrderF = false }
        "cla",       { pythonFunc = "plt.cla";          argTransform = Direct;         needsOrderF = false }
        "gcf",       { pythonFunc = "plt.gcf";          argTransform = Direct;         needsOrderF = false }
        "gca",       { pythonFunc = "plt.gca";          argTransform = Direct;         needsOrderF = false }
        "set",       { pythonFunc = "plt.setp";         argTransform = Direct;         needsOrderF = false }
        "get",       { pythonFunc = "plt.getp";         argTransform = Direct;         needsOrderF = false }
        "text",      { pythonFunc = "plt.text";         argTransform = Direct;         needsOrderF = false }
        "patch",     { pythonFunc = "plt.fill";         argTransform = Direct;         needsOrderF = false }
        "line",      { pythonFunc = "plt.plot";         argTransform = Direct;         needsOrderF = false }
        "fill",      { pythonFunc = "plt.fill";         argTransform = Direct;         needsOrderF = false }
        "quiver",    { pythonFunc = "plt.quiver";       argTransform = Direct;         needsOrderF = false }
        "clim",      { pythonFunc = "plt.clim";         argTransform = Direct;         needsOrderF = false }
        "xlim",      { pythonFunc = "plt.xlim";         argTransform = Direct;         needsOrderF = false }
        "ylim",      { pythonFunc = "plt.ylim";         argTransform = Direct;         needsOrderF = false }
        "zlim",      { pythonFunc = "plt.zlim";         argTransform = Direct;         needsOrderF = false }
        "view",      { pythonFunc = "plt.view";         argTransform = Direct;         needsOrderF = false }
        "shading",   { pythonFunc = "plt.shading";      argTransform = Direct;         needsOrderF = false }
        "saveas",    { pythonFunc = "plt.savefig";      argTransform = Direct;         needsOrderF = false }
        "print",     { pythonFunc = "plt.savefig";      argTransform = Direct;         needsOrderF = false }
        // I/O (additional)
        "fopen",     { pythonFunc = "open";             argTransform = Direct;         needsOrderF = false }
        "fclose",    { pythonFunc = "close";            argTransform = FileIoStyle "close"; needsOrderF = false }
        "fread",     { pythonFunc = "read";             argTransform = FileIoStyle "read"; needsOrderF = false }
        "fwrite",    { pythonFunc = "write";            argTransform = FileIoStyle "write"; needsOrderF = false }
        "fgets",     { pythonFunc = "readline";         argTransform = FileIoStyle "readline"; needsOrderF = false }
        "feof",      { pythonFunc = "";                 argTransform = Direct;         needsOrderF = false }
        "ftell",     { pythonFunc = "tell";             argTransform = FileIoStyle "tell"; needsOrderF = false }
        "fseek",     { pythonFunc = "seek";             argTransform = FileIoStyle "seek"; needsOrderF = false }
        // Matrix operations (additional)
        "chol",      { pythonFunc = "np.linalg.cholesky"; argTransform = Direct;       needsOrderF = false }
        "lu",        { pythonFunc = "scipy.linalg.lu";    argTransform = Direct;       needsOrderF = false }
        "qr",        { pythonFunc = "np.linalg.qr";      argTransform = Direct;       needsOrderF = false }
        "rank",      { pythonFunc = "np.linalg.matrix_rank"; argTransform = Direct;    needsOrderF = false }
        "cond",      { pythonFunc = "np.linalg.cond";    argTransform = Direct;        needsOrderF = false }
        "null",      { pythonFunc = "scipy.linalg.null_space"; argTransform = Direct;  needsOrderF = false }
        "orth",      { pythonFunc = "scipy.linalg.orth";  argTransform = Direct;       needsOrderF = false }
        "expm",      { pythonFunc = "scipy.linalg.expm";  argTransform = Direct;       needsOrderF = false }
        "logm",      { pythonFunc = "scipy.linalg.logm";  argTransform = Direct;       needsOrderF = false }
        "sqrtm",     { pythonFunc = "scipy.linalg.sqrtm"; argTransform = Direct;       needsOrderF = false }
        // Convolution
        "conv2",     { pythonFunc = "scipy.signal.convolve2d"; argTransform = Direct;  needsOrderF = false }
        "convn",     { pythonFunc = "scipy.ndimage.convolve";  argTransform = Direct;  needsOrderF = false }
        // Dynamic dispatch
        "feval",     { pythonFunc = "";                argTransform = FevalStyle;       needsOrderF = false }
        // Regex
        "regexp",    { pythonFunc = "re.search";       argTransform = RegexpStyle;      needsOrderF = false }
        "regexpi",   { pythonFunc = "re.search";       argTransform = RegexpStyle;      needsOrderF = false }
        "regexprep", { pythonFunc = "re.sub";          argTransform = RegexpRepStyle;   needsOrderF = false }
        // NaN constructors
        "nan",       { pythonFunc = "np.full";         argTransform = NanFullStyle;     needsOrderF = false }
        "NaN",       { pythonFunc = "np.full";         argTransform = NanFullStyle;     needsOrderF = false }
    ]

let tryMapBuiltin (name: string) : BuiltinMapping option =
    match builtinTable.TryGetValue(name) with
    | true, mapping -> Some mapping
    | false, _ -> None
