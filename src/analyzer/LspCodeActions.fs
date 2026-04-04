module LspCodeActions

open System.Text.RegularExpressions
open Ionide.LanguageServerProtocol.Types

// ---------------------------------------------------------------------------
// Quick fix code actions.
// ---------------------------------------------------------------------------

// Make a WorkspaceEdit replacing lineNum's text entirely with newText.
let private makeLineEdit (uri: string) (lineNum: int) (lineLen: int) (newText: string) : WorkspaceEdit =
    let range : Range = {
        Start = { Line = uint32 lineNum; Character = 0u }
        End   = { Line = uint32 lineNum; Character = uint32 lineLen }
    }
    let edit : TextEdit = { Range = range; NewText = newText }
    { Changes     = Some (Map.ofList [ uri, [| edit |] ])
      DocumentChanges = None
      ChangeAnnotations = None }

/// Generate quick-fix actions for a single LSP diagnostic.
let codeActionsForDiagnostic
    (diagnostic: Ionide.LanguageServerProtocol.Types.Diagnostic)
    (uri: string)
    (sourceLines: string array)
    : CodeAction array =

    let actions = ResizeArray<CodeAction>()

    // Diagnostic.Range.Start.Line is uint32 in Ionide
    let lineNum = int diagnostic.Range.Start.Line
    if lineNum < 0 || lineNum >= sourceLines.Length then
        actions.ToArray()
    else

    let lineText = sourceLines.[lineNum]
    let lineLen  = lineText.Length

    // Determine diagnostic code string (U2<int, string>)
    let codeStr =
        match diagnostic.Code with
        | Some (U2.C2 s) -> s
        | Some (U2.C1 n) -> string n
        | None -> ""

    // 1. W_INNER_DIM_MISMATCH with "elementwise multiplication" hint: * -> .*
    if codeStr = "W_INNER_DIM_MISMATCH" && diagnostic.Message.Contains("elementwise multiplication") then
        let newText = Regex.Replace(lineText, @"(?<!\.)\*", ".*")
        if newText <> lineText then
            let edit = makeLineEdit uri lineNum lineLen newText
            actions.Add({
                Title       = "Replace * with .* (elementwise)"
                Kind        = Some CodeActionKind.QuickFix
                Diagnostics = Some [| diagnostic |]
                Edit        = Some edit
                Command     = None
                IsPreferred = None
                Disabled    = None
                Data        = None
            })

    // 2. W_LOGICAL_OP_NON_SCALAR: && -> & and || -> |
    elif codeStr = "W_LOGICAL_OP_NON_SCALAR" then
        if lineText.Contains("&&") then
            let newText = lineText.Replace("&&", "&")
            let edit = makeLineEdit uri lineNum lineLen newText
            actions.Add({
                Title       = "Replace && with & (elementwise)"
                Kind        = Some CodeActionKind.QuickFix
                Diagnostics = Some [| diagnostic |]
                Edit        = Some edit
                Command     = None
                IsPreferred = None
                Disabled    = None
                Data        = None
            })
        if lineText.Contains("||") then
            let newText = lineText.Replace("||", "|")
            let edit = makeLineEdit uri lineNum lineLen newText
            actions.Add({
                Title       = "Replace || with | (elementwise)"
                Kind        = Some CodeActionKind.QuickFix
                Diagnostics = Some [| diagnostic |]
                Edit        = Some edit
                Command     = None
                IsPreferred = None
                Disabled    = None
                Data        = None
            })

    // 3. Suppress action: available for any Conformal diagnostic code
    if codeStr.StartsWith("W_") then
        let disableMarker = "% conformal:disable"
        let newText =
            let idx = lineText.IndexOf(disableMarker)
            if idx >= 0 then
                // Append the new code to the existing directive
                lineText + " " + codeStr
            else
                // Append a new disable comment
                lineText + "  " + disableMarker + " " + codeStr
        let edit = makeLineEdit uri lineNum lineLen newText
        actions.Add({
            Title       = $"Suppress {codeStr} on this line"
            Kind        = Some CodeActionKind.QuickFix
            Diagnostics = Some [| diagnostic |]
            Edit        = Some edit
            Command     = None
            IsPreferred = None
            Disabled    = None
            Data        = None
        })

    actions.ToArray()
