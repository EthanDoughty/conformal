module Workspace

open System
#if !FABLE_COMPILER
open System.IO
#endif
open System.Text.RegularExpressions
open Context

// ---------------------------------------------------------------------------
// Workspace scanning for multi-file MATLAB projects.
// ---------------------------------------------------------------------------

// Regex to extract first function signature from MATLAB source.
// Handles 3 forms:
//   function [a, b] = name(params)    # multi-return
//   function result = name(params)    # single-return
//   function name(params)             # procedure
let private funcSigPattern =
    @"^\s*function\s+(?:\[([^\]]*)\]\s*=\s*(\w+)|(\w+)\s*=\s*(\w+)|(\w+))\s*(?:\(([^)]*)\))?"

let private funcSigRegex =
    Regex(funcSigPattern, RegexOptions.Multiline)


/// Extract the first function signature from MATLAB source.
/// Returns (funcName, paramCount, returnCount) or None if no function found.
let extractFunctionSignature (source: string) : (string * int * int) option =
    let m = funcSigRegex.Match(source)
    if not m.Success then None
    else
        let multiRets  = m.Groups.[1].Value  // group 1: multi-return vars
        let multiName  = m.Groups.[2].Value  // group 2: multi-return func name
        let singleRet  = m.Groups.[3].Value  // group 3: single-return var
        let singleName = m.Groups.[4].Value  // group 4: single-return func name
        let procName   = m.Groups.[5].Value  // group 5: procedure func name
        let params_    = m.Groups.[6].Value  // group 6: parameter list

        let funcName, returnCount =
            if m.Groups.[1].Success then
                // Multi-return form: [a, b] = name(...)
                let returnVars =
                    multiRets.Split([| ','; ' '; '\t' |], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.filter (fun s -> s.Trim() <> "")
                (multiName, returnVars.Length)
            elif m.Groups.[3].Success then
                // Single-return form: result = name(...)
                (singleName, 1)
            else
                // Procedure form: name(...)
                (procName, 0)

        let paramCount =
            if not m.Groups.[6].Success || params_.Trim() = "" then 0
            else
                params_.Split(',')
                |> Array.filter (fun s -> s.Trim() <> "")
                |> Array.length

        Some (funcName, paramCount, returnCount)


#if !FABLE_COMPILER
// Matches a line starting with 'classdef' (after optional whitespace), ignoring comments.
let private classdefRegex =
    Regex(@"^\s*classdef\s+", RegexOptions.Multiline)

/// Scan a directory for .m files and extract function signatures.
/// Returns (externalFunctions map, externalClassdefs map).
/// maxDepth: levels of subdirectories to descend (0 = flat).
/// First-found wins on name collisions (shallower directory takes priority).
/// Hidden directories (starting with '.') and 'private/' are skipped at depth > 0.
let scanWorkspace (dirPath: string) (excludeFile: string) (maxDepth: int)
    : Map<string, ExternalSignature> * Map<string, string> =
    let funcResult  = System.Collections.Generic.Dictionary<string, ExternalSignature>()
    let classResult = System.Collections.Generic.Dictionary<string, string>()

    if not (Directory.Exists(dirPath)) then Map.empty, Map.empty
    else
        let excludeFullPath =
            if excludeFile <> "" then Path.GetFullPath(Path.Combine(dirPath, excludeFile)) else ""

        let rec scanDir (dir: string) (depth: int) =
            if depth > maxDepth then ()
            else
                let dirName = Path.GetFileName(dir)
                // Skip hidden dirs (except root at depth 0) and private/ (reserved for P1)
                if depth > 0 && (dirName.StartsWith(".") || dirName = "private") then ()
                else
                    // Process files at this level first; sort for determinism
                    let mFiles = Directory.GetFiles(dir, "*.m") |> Array.sort
                    for filePath in mFiles do
                        let fullPath = Path.GetFullPath(filePath)
                        if fullPath <> excludeFullPath then
                            try
                                let source = File.ReadAllText(filePath)
                                let key = Path.GetFileNameWithoutExtension(filePath)
                                // First-found wins: shallower directory takes priority
                                if not (funcResult.ContainsKey(key)) && not (classResult.ContainsKey(key)) then
                                    if classdefRegex.IsMatch(source) then
                                        classResult.[key] <- filePath
                                    else
                                        match extractFunctionSignature source with
                                        | None -> ()  // Script file or no function found
                                        | Some (funcName, paramCount, returnCount) ->
                                            funcResult.[key] <- {
                                                filename    = funcName
                                                paramCount  = paramCount
                                                returnCount = returnCount
                                                sourcePath  = filePath
                                                body        = None
                                                parmNames   = []
                                                outputNames = []
                                            }
                            with _ -> ()  // File read failed; skip silently
                    // Recurse into subdirectories; sort for determinism
                    let subDirs = Directory.GetDirectories(dir) |> Array.sort
                    for subDir in subDirs do
                        scanDir subDir (depth + 1)

        scanDir dirPath 0
        let funcMap  = funcResult  |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Map.ofSeq
        let classMap = classResult |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Map.ofSeq
        funcMap, classMap


/// Scan <dirPath>/private/*.m and return a map of ExternalSignature.
/// Returns Map.empty if the private/ directory does not exist.
/// Private functions are visible only to callers in the parent directory (MATLAB convention).
let scanPrivateDir (dirPath: string) : Map<string, ExternalSignature> =
    let privatePath = Path.Combine(dirPath, "private")
    if not (Directory.Exists(privatePath)) then Map.empty
    else
        let result = System.Collections.Generic.Dictionary<string, ExternalSignature>()
        let mFiles = Directory.GetFiles(privatePath, "*.m") |> Array.sort
        for filePath in mFiles do
            try
                let source = File.ReadAllText(filePath)
                match extractFunctionSignature source with
                | None -> ()
                | Some (funcName, paramCount, returnCount) ->
                    let key = Path.GetFileNameWithoutExtension(filePath)
                    result.[key] <- {
                        filename    = funcName
                        paramCount  = paramCount
                        returnCount = returnCount
                        sourcePath  = filePath
                        body        = None
                        parmNames   = []
                        outputNames = []
                    }
            with _ -> ()
        result |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Map.ofSeq
#endif


// ---------------------------------------------------------------------------
// Parse cache: path -> (content_hash, primary_sig * subfunctions)
// ---------------------------------------------------------------------------

let private parsedCache =
    System.Collections.Generic.Dictionary<string, string * (FunctionSignature * Map<string, FunctionSignature>)>()


/// Clear the parsed external file cache. Callers must invoke between test runs to avoid stale parses.
let clearParseCache () =
    parsedCache.Clear()


#if !FABLE_COMPILER
// Compute MD5 hash of a string.
let private contentHash (content: string) : string =
    use md5 = System.Security.Cryptography.MD5.Create()
    let bytes = System.Text.Encoding.UTF8.GetBytes(content)
    let hash = md5.ComputeHash(bytes)
    hash |> Array.map (fun b -> b.ToString("x2")) |> String.concat ""


// .NET-only path (Fable uses pre-parsed IR). Load and parse an external .m file.
let loadExternalFunctionFromPath
    (sourcePath: string)
    (parseAndBuildIr: string -> (FunctionSignature * Map<string, FunctionSignature>) option)
    : (FunctionSignature * Map<string, FunctionSignature>) option =

    try
        let source = File.ReadAllText(sourcePath)
        let hash = contentHash source

        match parsedCache.TryGetValue(sourcePath) with
        | true, (cachedHash, result) when cachedHash = hash -> Some result
        | _ ->
            match parseAndBuildIr source with
            | None -> None
            | Some result ->
                parsedCache.[sourcePath] <- (hash, result)
                Some result
    with _ -> None
#endif


/// Parse MATLAB source and extract function signatures.
/// Returns (primary_FunctionSignature, subfunctions_dict) or None if no function found.
let buildIrFromSource (source: string) : (FunctionSignature * Map<string, FunctionSignature>) option =
    try
        let (program, _) = Parser.parseMATLAB source
        let funcDefs =
            program.body
            |> List.choose (fun stmt ->
                match stmt with
                | Ir.FunctionDef({ line = line; col = col }, name, parms, outputVars, body, argAnns) ->
                    Some { name = name; parms = parms; outputVars = outputVars; body = body
                           defLine = line; defCol = col; argShapes = Shapes.argAnnotationsToShapes argAnns }
                | _ -> None)
        match funcDefs with
        | [] -> None
        | primary :: rest ->
            let subfunctions =
                rest |> List.map (fun s -> (s.name, s)) |> Map.ofList
            Some (primary, subfunctions)
    with _ -> None


#if FABLE_COMPILER
/// Fable path: uses pre-parsed body from Interop.
let loadExternalFunction (sig_: ExternalSignature) : (FunctionSignature * Map<string, FunctionSignature>) option =
    match sig_.body with
    | Some body ->
        let primary = { name = sig_.filename; parms = sig_.parmNames; outputVars = sig_.outputNames; body = body; defLine = 1; defCol = 0; argShapes = Map.empty }
        Some (primary, Map.empty)
    | None -> None
#else
/// Load and parse an external .m file.
/// Returns (primary_FunctionSignature, subfunctions_dict) or None on error.
let loadExternalFunction (sig_: ExternalSignature) : (FunctionSignature * Map<string, FunctionSignature>) option =
    if sig_.sourcePath = "" then None
    else loadExternalFunctionFromPath sig_.sourcePath buildIrFromSource


/// Parse a classdef .m file and extract ClassInfo + method signatures.
/// Returns (className, propNames, methodSigs, superName) or None if the file cannot be parsed
/// or does not contain a classdef OpaqueStmt.
let loadExternalClassdef (sourcePath: string)
    : (string * string list * Map<string, FunctionSignature> * string option) option =
    try
        let source = File.ReadAllText(sourcePath)
        let (program, _) = Parser.parseMATLAB source
        // Collect all FunctionDef nodes (methods)
        let methodSigs =
            program.body
            |> List.choose (fun stmt ->
                match stmt with
                | Ir.FunctionDef({ line = line; col = col }, name, parms, outputVars, body, argAnns) ->
                    Some (name, { name = name; parms = parms; outputVars = outputVars
                                  body = body; defLine = line; defCol = col
                                  argShapes = Shapes.argAnnotationsToShapes argAnns })
                | _ -> None)
            |> Map.ofList
        // Find the OpaqueStmt that encodes classdef metadata
        let classdefOpaqueOpt =
            program.body
            |> List.tryPick (fun stmt ->
                match stmt with
                | Ir.OpaqueStmt(_, _, raw) when raw.StartsWith("classdef:") -> Some raw
                | _ -> None)
        match classdefOpaqueOpt with
        | None -> None
        | Some raw ->
            let parts = raw.Split(':')
            if parts.Length < 2 then None
            else
                let className = parts.[1]
                let propNames =
                    if parts.Length >= 3 && parts.[2] <> "" then
                        parts.[2].Split(',') |> Array.toList
                    else []
                let superName =
                    if parts.Length >= 4 && parts.[3] <> "" then Some parts.[3] else None
                Some (className, propNames, methodSigs, superName)
    with _ -> None
#endif
