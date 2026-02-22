module Context

open Ir
open Shapes
open Env
open PathConstraints

// ---------------------------------------------------------------------------
// Function signatures
// ---------------------------------------------------------------------------

type FunctionSignature = {
    name:       string
    parms:      string list
    outputVars: string list
    body:       Stmt list
}

/// Signature extracted from an external .m file.
type ExternalSignature = {
    filename:    string
    paramCount:  int
    returnCount: int
    /// Full path to the source .m file (used by loadExternalFunction)
    sourcePath:  string
    /// Parsed body (None if not yet parsed or parse failed)
    body:        Stmt list option
    /// Names of parameters
    parmNames:   string list
    /// Names of output variables
    outputNames: string list
}

// ---------------------------------------------------------------------------
// Control-flow exceptions (mirrors Python EarlyReturn / EarlyBreak / EarlyContinue)
// ---------------------------------------------------------------------------

exception EarlyReturn
exception EarlyBreak
exception EarlyContinue

// ---------------------------------------------------------------------------
// Sub-contexts
// ---------------------------------------------------------------------------

type CallContext() =
    /// Maps function name -> FunctionSignature (same-file user functions)
    member val functionRegistry       : System.Collections.Generic.Dictionary<string, FunctionSignature>
                                        = System.Collections.Generic.Dictionary<string, FunctionSignature>() with get, set
    /// Maps function name -> FunctionSignature (nested/local functions)
    member val nestedFunctionRegistry : System.Collections.Generic.Dictionary<string, FunctionSignature>
                                        = System.Collections.Generic.Dictionary<string, FunctionSignature>() with get, set
    /// Set of function names currently being analyzed (recursion guard)
    member val analyzingFunctions     : System.Collections.Generic.HashSet<string>
                                        = System.Collections.Generic.HashSet<string>() with get, set
    /// Set of lambda IDs currently being analyzed (recursion guard)
    member val analyzingLambdas       : System.Collections.Generic.HashSet<int>
                                        = System.Collections.Generic.HashSet<int>() with get, set
    /// Set of external function names currently being analyzed (cycle guard)
    member val analyzingExternal      : System.Collections.Generic.HashSet<string>
                                        = System.Collections.Generic.HashSet<string>() with get, set
    /// Cache keyed by (funcName, argShapes tuple string) -> (outputShapes, warnings)
    member val analysisCache          : System.Collections.Generic.Dictionary<string, obj>
                                        = System.Collections.Generic.Dictionary<string, obj>() with get, set
    /// Lambda metadata: lambda_id -> (params, body_expr, closure_env)
    member val lambdaMetadata         : System.Collections.Generic.Dictionary<int, obj>
                                        = System.Collections.Generic.Dictionary<int, obj>() with get, set
    /// Handle registry: handle_id -> function_name
    member val handleRegistry         : System.Collections.Generic.Dictionary<int, string>
                                        = System.Collections.Generic.Dictionary<int, string>() with get, set
    /// Counter for allocating fresh lambda IDs
    member val nextLambdaId   : int = 0 with get, set
    /// Enable fixpoint-based loop analysis
    member val fixpoint       : bool = false with get, set
    /// Current call depth (for recursion limit)
    member val callDepth      : int = 0 with get, set

type ConstraintContext() =
    /// Set of (dim1_str, dim2_str) equality constraints (canonicalized)
    member val constraints         : System.Collections.Generic.HashSet<string * string>
                                     = System.Collections.Generic.HashSet<string * string>() with get, set
    /// Provenance: (dim1_str, dim2_str) -> source_line
    member val constraintProvenance : System.Collections.Generic.Dictionary<string * string, int>
                                      = System.Collections.Generic.Dictionary<string * string, int>() with get, set
    /// Concrete bindings: var_name -> concrete_value
    member val scalarBindings      : System.Collections.Generic.Dictionary<string, int>
                                     = System.Collections.Generic.Dictionary<string, int>() with get, set
    /// Value ranges: var_name -> interval (stored as obj until Intervals.fs is defined)
    member val valueRanges         : System.Collections.Generic.Dictionary<string, obj>
                                     = System.Collections.Generic.Dictionary<string, obj>() with get, set
    /// Conflict sites accumulated globally (stored as obj list until Witness.fs is defined)
    member val conflictSites : obj list = [] with get, set
    /// Dim provenance: (var_name, "rows"|"cols") -> Dim
    member val dimProvenance       : System.Collections.Generic.Dictionary<string * string, Dim>
                                     = System.Collections.Generic.Dictionary<string * string, Dim>() with get, set
    /// Branch path constraint stack
    member val pathConstraints     : PathConstraintStack = PathConstraintStack() with get, set
    /// Colon context (used to track when ':' is in a subscript context)
    member val colonContext : bool = false with get, set
    /// Strict mode (show all warnings)
    member val strictMode   : bool = false with get, set

type WorkspaceContext() =
    /// Maps function name -> ExternalSignature (workspace-scanned)
    member val externalFunctions : System.Collections.Generic.Dictionary<string, ExternalSignature>
                                   = System.Collections.Generic.Dictionary<string, ExternalSignature>() with get, set
    /// Working directory for workspace scanning
    member val workspaceDir : string = "" with get, set
    /// Set of external function names currently being analyzed (cross-file cycle guard)
    member val analyzingExternal   : System.Collections.Generic.HashSet<string>
                                     = System.Collections.Generic.HashSet<string>() with get, set

// ---------------------------------------------------------------------------
// AnalysisContext: root container for all analysis state
// ---------------------------------------------------------------------------

type AnalysisContext() =
    member val call : CallContext        = CallContext()       with get
    member val cst  : ConstraintContext  = ConstraintContext() with get
    member val ws   : WorkspaceContext   = WorkspaceContext()  with get
    /// Accumulated diagnostics
    member val diagnostics : Diagnostics.Diagnostic list = [] with get, set

    /// Save and restore scope-sensitive fields around function/lambda body analysis.
    /// Mirrors Python's snapshot_scope() context manager.
    /// Usage: ctx.SnapshotScope(fun () -> analyzeBody())
    member this.SnapshotScope (body: unit -> 'T) : 'T =
        // Save
        let savedConstraints  = System.Collections.Generic.HashSet<string * string>(this.cst.constraints)
        let savedProvenance   = System.Collections.Generic.Dictionary<string * string, int>(this.cst.constraintProvenance)
        let savedScalars      = System.Collections.Generic.Dictionary<string, int>(this.cst.scalarBindings)
        let savedRanges       = System.Collections.Generic.Dictionary<string, obj>(this.cst.valueRanges)
        let savedNested       = System.Collections.Generic.Dictionary<string, FunctionSignature>(this.call.nestedFunctionRegistry)
        try
            body ()
        finally
            // Restore
            this.cst.constraints.Clear()
            for kv in savedConstraints do this.cst.constraints.Add(kv) |> ignore
            this.cst.constraintProvenance.Clear()
            for kv in savedProvenance do this.cst.constraintProvenance.[kv.Key] <- kv.Value
            this.cst.scalarBindings.Clear()
            for kv in savedScalars do this.cst.scalarBindings.[kv.Key] <- kv.Value
            this.cst.valueRanges.Clear()
            for kv in savedRanges do this.cst.valueRanges.[kv.Key] <- kv.Value
            this.call.nestedFunctionRegistry.Clear()
            for kv in savedNested do this.call.nestedFunctionRegistry.[kv.Key] <- kv.Value

// ---------------------------------------------------------------------------
// BuiltinEvalContext: callback record to break EvalExpr <-> EvalBuiltins
// circular dependency (populated in Phase 3/4).
// ---------------------------------------------------------------------------

type BuiltinEvalContext = {
    /// evalExprIr: Expr -> Env -> AnalysisContext -> Shape
    evalExprIr         : Expr -> Env -> AnalysisContext -> Shapes.Shape
    /// exprToDimIr: Expr -> Env -> Dim
    exprToDimIr        : Expr -> Env -> Shapes.Dim
    /// exprToDimWithEnd: Expr -> Env -> Dim -> Dim  (End substitution)
    exprToDimWithEnd   : Expr -> Env -> Shapes.Dim -> Shapes.Dim
    /// getExprInterval: Expr -> Env -> AnalysisContext -> obj option
    getExprInterval    : Expr -> Env -> AnalysisContext -> obj option
    /// getConcreteDimSize: Dim -> AnalysisContext -> int option
    getConcreteDimSize : Shapes.Dim -> AnalysisContext -> int option
    /// unwrapArg: IndexArg -> Expr option  (extract scalar expr from IndexExpr)
    unwrapArg          : IndexArg -> Expr option
}
