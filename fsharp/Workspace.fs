module Workspace

open System
open System.IO
open System.Text.RegularExpressions
open Context

// ---------------------------------------------------------------------------
// Workspace scanning for multi-file MATLAB projects.
// Port of analysis/workspace.py
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


/// extractFunctionSignature: extract the first function signature from MATLAB source.
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


/// scanWorkspace: scan a directory for .m files and extract function signatures.
let scanWorkspace (dirPath: string) (excludeFile: string) : Map<string, ExternalSignature> =
    let result = System.Collections.Generic.Dictionary<string, ExternalSignature>()

    if not (Directory.Exists(dirPath)) then Map.empty
    else
        let mFiles = Directory.GetFiles(dirPath, "*.m")
        for filePath in mFiles do
            let fileName = Path.GetFileName(filePath)
            if fileName <> excludeFile then
                try
                    let source = File.ReadAllText(filePath)
                    match extractFunctionSignature source with
                    | None -> ()  // Script file or no function found
                    | Some (funcName, paramCount, returnCount) ->
                        let key = Path.GetFileNameWithoutExtension(filePath)
                        result.[key] <- {
                            filename    = funcName
                            paramCount  = paramCount
                            returnCount = returnCount
                            body        = None
                            parmNames   = []
                            outputNames = []
                        }
                with _ -> ()  // File read failed; skip silently

        result |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Map.ofSeq


// ---------------------------------------------------------------------------
// Parse cache: path -> (content_hash, primary_sig * subfunctions)
// ---------------------------------------------------------------------------

let private parsedCache =
    System.Collections.Generic.Dictionary<string, string * (FunctionSignature * Map<string, FunctionSignature>)>()


/// clearParseCache: clear the parsed external file cache.
let clearParseCache () =
    parsedCache.Clear()


/// contentHash: compute MD5 hash of a string.
let private contentHash (content: string) : string =
    use md5 = System.Security.Cryptography.MD5.Create()
    let bytes = System.Text.Encoding.UTF8.GetBytes(content)
    let hash = md5.ComputeHash(bytes)
    hash |> Array.map (fun b -> b.ToString("x2")) |> String.concat ""


/// loadExternalFunction: load and parse an external .m file.
/// Returns (primary_FunctionSignature, subfunctions_dict) or None on error.
let loadExternalFunction (sig_: ExternalSignature) : (FunctionSignature * Map<string, FunctionSignature>) option =
    // We need the file path to load from. ExternalSignature.filename contains the function name.
    // The source_path equivalent is stored in filename for the F# port.
    // However ExternalSignature.filename is the function name, not the path.
    // We use a workaround: look up the actual file path from the workspace scan context.
    // For Phase 4, we derive the path from the workspace dir + fname + ".m".
    // Since ExternalSignature doesn't have source_path in this F# port (unlike Python),
    // we store the full path in the filename field during scanWorkspace.
    // Actually the Python version has source_path; F# ExternalSignature has filename (func name) only.
    // We need to thread the path through. Use outputNames field to store path as workaround,
    // OR: use a separate module-level dict. For now, use the parmNames.[0] as file path if it starts with '/'.
    //
    // Better approach: the caller should pass the file path. We'll store it in ExternalSignature.filename
    // but scan uses key = stem, value has filename = funcName. We need the actual file path.
    // Let's store the actual source path in parmNames.[0] as a sentinel during scanning.
    // This is a temporary bridge approach.

    // For now, use a simple approach: the parmNames is used to store the path if set by the richer scan.
    // Fall back to None if we can't locate the file.
    None  // Stub: full implementation requires Parser integration (Phase 5)


/// loadExternalFunctionFromPath: load and parse an external .m file given its full path.
/// This is the real implementation; called from analysis when we have the path.
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
