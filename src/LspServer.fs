module LspServer

open System
open System.IO
open System.Security.Cryptography
open System.Text
open System.Threading
open Ionide.LanguageServerProtocol
open Ionide.LanguageServerProtocol.Types
open Ionide.LanguageServerProtocol.Server
open Ionide.LanguageServerProtocol.JsonUtils
open Newtonsoft.Json
open StreamJsonRpc
open Context
open Diagnostics
open Analysis
open Workspace
open Witness
open Builtins

// ---------------------------------------------------------------------------
// Analysis cache per document URI
// ---------------------------------------------------------------------------

type AnalysisCache = {
    mutable env:               Env.Env
    mutable diagnostics:       Diagnostics.Diagnostic list
    mutable sourceHash:        string
    mutable settingsHash:      string
    mutable irProg:            Ir.Program option
    mutable functionRegistry:  System.Collections.Generic.Dictionary<string, FunctionSignature>
    mutable externalFunctions: System.Collections.Generic.Dictionary<string, ExternalSignature>
}

// ---------------------------------------------------------------------------
// Server settings
// ---------------------------------------------------------------------------

type ServerSettings() =
    member val fixpoint      : bool = false with get, set
    member val strict        : bool = false with get, set
    member val analyzeOnChange : bool = false with get, set

// ---------------------------------------------------------------------------
// ConformalClient: sends notifications/requests to the IDE
// ---------------------------------------------------------------------------

type ConformalClient(notifier: ClientNotificationSender, _requester: ClientRequestSender) =
    inherit LspClient()

    override _.TextDocumentPublishDiagnostics(p: PublishDiagnosticsParams) : Async<unit> =
        notifier "textDocument/publishDiagnostics" (box p) |> Async.Ignore

    override _.WindowShowMessage(p: ShowMessageParams) : Async<unit> =
        notifier "window/showMessage" (box p) |> Async.Ignore

    override _.WindowLogMessage(p: LogMessageParams) : Async<unit> =
        notifier "window/logMessage" (box p) |> Async.Ignore

// ---------------------------------------------------------------------------
// ConformalLspServer: handles requests from the IDE
// ---------------------------------------------------------------------------

type ConformalLspServer(client: ConformalClient) =
    inherit LspServer()

    // Per-document analysis cache: URI -> AnalysisCache
    let analysisCache = System.Collections.Generic.Dictionary<string, AnalysisCache>()

    // Debouncing: URI -> CancellationTokenSource
    let debounceTokens = System.Collections.Generic.Dictionary<string, CancellationTokenSource>()

    // Server settings
    let settings = ServerSettings()

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    let computeHash (text: string) : string =
        use sha256 = SHA256.Create()
        let bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(text))
        BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant()

    let computeSettingsHash () : string =
        computeHash (string settings.fixpoint + string settings.strict)

    let uriToPath (uri: string) : string =
        try
            let u = Uri(uri)
            u.LocalPath
        with _ ->
            if uri.StartsWith("file://") then
                Uri.UnescapeDataString(uri.Substring(7))
            else uri

    let publishDiagnostics (uri: string) (lspDiags: Ionide.LanguageServerProtocol.Types.Diagnostic array) =
        client.TextDocumentPublishDiagnostics({
            Uri         = uri
            Diagnostics = lspDiags
            Version     = None
        }) |> Async.RunSynchronously

    let logErr (msg: string) =
        Console.Error.WriteLine("[conformal-lsp] " + msg)

    // ---------------------------------------------------------------------------
    // _validate: parse + analyze + publish diagnostics
    // ---------------------------------------------------------------------------

    let validate (uri: string) (source: string) (force: bool) =
        let sourceHash   = computeHash source
        let settingsHash = computeSettingsHash ()
        let sourceLines  = source.Split('\n')

        // Cache skip if source and settings unchanged (unless forced)
        if not force && analysisCache.ContainsKey(uri) then
            let cached = analysisCache.[uri]
            if cached.sourceHash = sourceHash && cached.settingsHash = settingsHash then
                // Re-publish cached diagnostics (no witness on cache replay)
                let lspDiags =
                    cached.diagnostics
                    |> List.map (fun d -> LspDiagnostics.toLspDiagnostic d sourceLines uri None)
                    |> Array.ofList
                publishDiagnostics uri lspDiags

        else

        try
            // Scan workspace for external functions
            let filePath = uriToPath uri
            let dirPath  = Path.GetDirectoryName(Path.GetFullPath(filePath))
            let fileName = Path.GetFileName(filePath)
            let extMap   = scanWorkspace dirPath fileName

            // Build analysis context
            let ctx = AnalysisContext()
            ctx.call.fixpoint <- settings.fixpoint
            for kv in extMap do
                ctx.ws.externalFunctions.[kv.Key] <- kv.Value
            ctx.ws.workspaceDir <- dirPath

            // Parse and analyze
            let irProg = Parser.parseMATLAB source
            let (env, warnings) = analyzeProgramIr irProg ctx

            // Filter strict-only in default mode
            let filteredWarnings =
                if settings.strict then warnings
                else warnings |> List.filter (fun w -> not (Set.contains w.code STRICT_ONLY_CODES))

            // Generate witnesses from conflict sites (stored as obj list, cast to ConflictSite list)
            let conflictSites =
                ctx.cst.conflictSites
                |> List.choose (fun o ->
                    try Some (o :?> ConflictSite) with _ -> None)
            let witnesses = generateWitnesses conflictSites

            // Convert to LSP diagnostics
            let lspDiags =
                filteredWarnings
                |> List.map (fun d ->
                    let key = (d.line, d.code)
                    let wit = Map.tryFind key witnesses
                    LspDiagnostics.toLspDiagnostic d sourceLines uri wit)
                |> Array.ofList

            publishDiagnostics uri lspDiags

            // Update cache
            let cache = {
                env               = env
                diagnostics       = filteredWarnings
                sourceHash        = sourceHash
                settingsHash      = settingsHash
                irProg            = Some irProg
                functionRegistry  = ctx.call.functionRegistry
                externalFunctions = ctx.ws.externalFunctions
            }
            analysisCache.[uri] <- cache

        with
        | Parser.ParseError msg ->
            let errorDiag : Ionide.LanguageServerProtocol.Types.Diagnostic = {
                Range    = { Start = { Line = 0u; Character = 0u }; End = { Line = 0u; Character = 0u } }
                Severity = Some DiagnosticSeverity.Error
                Code     = None
                Source   = Some "conformal"
                Message  = "Syntax error: " + msg
                Tags     = None
                RelatedInformation = None
                CodeDescription    = None
                Data               = None
            }
            publishDiagnostics uri [| errorDiag |]
        | Lexer.LexError msg ->
            let errorDiag : Ionide.LanguageServerProtocol.Types.Diagnostic = {
                Range    = { Start = { Line = 0u; Character = 0u }; End = { Line = 0u; Character = 0u } }
                Severity = Some DiagnosticSeverity.Error
                Code     = None
                Source   = Some "conformal"
                Message  = "Syntax error: " + msg
                Tags     = None
                RelatedInformation = None
                CodeDescription    = None
                Data               = None
            }
            publishDiagnostics uri [| errorDiag |]
        | ex ->
            let errorDiag : Ionide.LanguageServerProtocol.Types.Diagnostic = {
                Range    = { Start = { Line = 0u; Character = 0u }; End = { Line = 0u; Character = 0u } }
                Severity = Some DiagnosticSeverity.Error
                Code     = None
                Source   = Some "conformal"
                Message  = "Internal error: " + ex.Message
                Tags     = None
                RelatedInformation = None
                CodeDescription    = None
                Data               = None
            }
            publishDiagnostics uri [| errorDiag |]
            logErr ("Analysis error for " + uri + ": " + ex.Message)

    // ---------------------------------------------------------------------------
    // LspServer overrides
    // ---------------------------------------------------------------------------

    override _.Initialize(p: InitializeParams) = async {
        // Read initializationOptions (may be a JObject from JSON-RPC deserialization)
        match p.InitializationOptions with
        | Some opts ->
            try
                let jobj = opts :?> Newtonsoft.Json.Linq.JObject
                match jobj.TryGetValue("fixpoint") with
                | true, v -> settings.fixpoint <- v.Value<bool>()
                | _ -> ()
                match jobj.TryGetValue("strict") with
                | true, v -> settings.strict <- v.Value<bool>()
                | _ -> ()
                match jobj.TryGetValue("analyzeOnChange") with
                | true, v -> settings.analyzeOnChange <- v.Value<bool>()
                | _ -> ()
            with _ -> ()
        | None -> ()

        let syncOpts : TextDocumentSyncOptions = {
            OpenClose         = Some true
            Change            = Some TextDocumentSyncKind.Full
            WillSave          = None
            WillSaveWaitUntil = None
            Save              = Some (U2.C2 { IncludeText = Some true })
        }

        let capabilities : ServerCapabilities = {
            TextDocumentSync          = Some (U2.C1 syncOpts)
            HoverProvider             = Some (U2.C1 true)
            DocumentSymbolProvider    = Some (U2.C1 true)
            CodeActionProvider        = Some (U2.C2 {
                CodeActionKinds  = Some [| CodeActionKind.QuickFix |]
                ResolveProvider  = None
                WorkDoneProgress = None
            })
            CompletionProvider        = None
            SignatureHelpProvider      = None
            DefinitionProvider        = None
            ReferencesProvider        = None
            DocumentHighlightProvider = None
            DocumentFormattingProvider = None
            DocumentRangeFormattingProvider = None
            DocumentOnTypeFormattingProvider = None
            RenameProvider            = None
            DocumentLinkProvider      = None
            ExecuteCommandProvider    = None
            WorkspaceSymbolProvider   = None
            DeclarationProvider       = None
            TypeDefinitionProvider    = None
            ImplementationProvider    = None
            ColorProvider             = None
            FoldingRangeProvider      = None
            SelectionRangeProvider    = None
            LinkedEditingRangeProvider = None
            CallHierarchyProvider     = None
            SemanticTokensProvider    = None
            MonikerProvider           = None
            TypeHierarchyProvider     = None
            InlineValueProvider       = None
            InlayHintProvider         = None
            DiagnosticProvider        = None
            Workspace                 = None
            Experimental              = None
            PositionEncoding          = None
            NotebookDocumentSync      = None
            CodeLensProvider          = None
        }

        return Ok {
            Capabilities = capabilities
            ServerInfo   = Some { Name = "conformal"; Version = Some "1.0" }
        }
    }

    override _.TextDocumentDidOpen(p: DidOpenTextDocumentParams) = async {
        validate p.TextDocument.Uri p.TextDocument.Text false
    }

    override _.TextDocumentDidSave(p: DidSaveTextDocumentParams) = async {
        // Get source from the save notification (if included) or read from disk
        let source =
            match p.Text with
            | Some t -> t
            | None ->
                let filePath = uriToPath p.TextDocument.Uri
                if File.Exists(filePath) then
                    try File.ReadAllText(filePath) with _ -> ""
                else ""

        if source <> "" then
            validate p.TextDocument.Uri source false

        // Cross-file invalidation: re-analyze sibling .m files
        let savedPath = uriToPath p.TextDocument.Uri
        let savedDir  =
            try Path.GetDirectoryName(Path.GetFullPath(savedPath))
            with _ -> ""

        if savedDir <> "" then
            for kv in (analysisCache |> Seq.toList) do
                let cachedUri = kv.Key
                if cachedUri <> p.TextDocument.Uri then
                    try
                        let cachedPath = uriToPath cachedUri
                        let cachedDir  = Path.GetDirectoryName(Path.GetFullPath(cachedPath))
                        if cachedDir = savedDir && File.Exists(cachedPath) then
                            let otherSource = File.ReadAllText(cachedPath)
                            validate cachedUri otherSource true
                    with _ -> ()
    }

    override _.TextDocumentDidChange(p: DidChangeTextDocumentParams) = async {
        if not settings.analyzeOnChange then ()
        else

        let uri = p.TextDocument.Uri

        // Cancel any pending debounce
        match debounceTokens.TryGetValue(uri) with
        | true, cts ->
            cts.Cancel()
            debounceTokens.Remove(uri) |> ignore
        | _ -> ()

        // Get latest text from changes (ContentChanges is U2<C1 (incremental), C2 (full)>[])
        let latestText =
            p.ContentChanges
            |> Array.tryLast
            |> Option.map (fun change ->
                match change with
                | U2.C1 c1 -> c1.Text   // partial change (incremental)
                | U2.C2 c2 -> c2.Text)  // full document text

        match latestText with
        | None -> ()
        | Some text ->
            // Schedule debounced validation after 500ms
            let cts = new CancellationTokenSource()
            debounceTokens.[uri] <- cts
            let token = cts.Token

            let _task =
                async {
                    do! Async.Sleep 500
                    if not token.IsCancellationRequested then
                        validate uri text false
                        match debounceTokens.TryGetValue(uri) with
                        | true, currentCts when obj.ReferenceEquals(currentCts, cts) ->
                            debounceTokens.Remove(uri) |> ignore
                        | _ -> ()
                } |> Async.StartAsTask
            ()
    }

    override _.TextDocumentHover(p: HoverParams) = async {
        let uri = p.TextDocument.Uri
        match analysisCache.TryGetValue(uri) with
        | false, _ -> return Ok None
        | true, cached ->
            // Read source from disk for hover context
            let source =
                try
                    let filePath = uriToPath uri
                    if File.Exists(filePath) then File.ReadAllText(filePath) else ""
                with _ -> ""

            if source = "" then return Ok None
            else

            let hover =
                LspHover.getHover
                    cached.env
                    source
                    (int p.Position.Line)
                    (int p.Position.Character)
                    cached.functionRegistry
                    KNOWN_BUILTINS
                    cached.externalFunctions

            return Ok hover
    }

    override _.TextDocumentCodeAction(p: CodeActionParams) = async {
        let uri = p.TextDocument.Uri
        let source =
            try
                let filePath = uriToPath uri
                if File.Exists(filePath) then File.ReadAllText(filePath) else ""
            with _ -> ""

        if source = "" then return Ok None
        else

        let sourceLines = source.Split('\n')
        let allActions : U2<Command, CodeAction> array =
            p.Context.Diagnostics
            |> Array.collect (fun d -> LspCodeActions.codeActionsForDiagnostic d uri sourceLines)
            |> Array.map U2.C2

        if allActions.Length = 0 then return Ok None
        else return Ok (Some allActions)
    }

    override _.TextDocumentDocumentSymbol(p: DocumentSymbolParams) = async {
        let uri = p.TextDocument.Uri
        match analysisCache.TryGetValue(uri) with
        | false, _ -> return Ok None
        | true, cached ->
            match cached.irProg with
            | None -> return Ok None
            | Some irProg ->
                let source =
                    try
                        let filePath = uriToPath uri
                        if File.Exists(filePath) then File.ReadAllText(filePath) else ""
                    with _ -> ""

                let sourceLines = source.Split('\n')
                let symbols = LspSymbols.getDocumentSymbols irProg sourceLines
                if symbols.Length = 0 then return Ok None
                else
                    // TextDocumentDocumentSymbol returns Option<U2<SymbolInformation[], DocumentSymbol[]>>
                    // We return DocumentSymbol[] wrapped in U2.C2
                    return Ok (Some (U2.C2 symbols))
    }

    override _.WorkspaceDidChangeConfiguration(p: DidChangeConfigurationParams) = async {
        // Try to extract conformal settings from the configuration object
        try
            let jobj = p.Settings :?> Newtonsoft.Json.Linq.JObject
            let conformal =
                match jobj.TryGetValue("conformal") with
                | true, v -> v :?> Newtonsoft.Json.Linq.JObject
                | _ -> jobj
            match conformal.TryGetValue("fixpoint") with
            | true, v -> settings.fixpoint <- v.Value<bool>()
            | _ -> ()
            match conformal.TryGetValue("strict") with
            | true, v -> settings.strict <- v.Value<bool>()
            | _ -> ()
            match conformal.TryGetValue("analyzeOnChange") with
            | true, v -> settings.analyzeOnChange <- v.Value<bool>()
            | _ -> ()
        with _ -> ()

        // Re-analyze all cached documents with new settings
        for kv in (analysisCache |> Seq.toList) do
            try
                let filePath = uriToPath kv.Key
                if File.Exists(filePath) then
                    let source = File.ReadAllText(filePath)
                    validate kv.Key source true
            with _ -> ()
    }

    override _.Dispose() = ()

// ---------------------------------------------------------------------------
// Bootstrap: start the LSP server
// ---------------------------------------------------------------------------

let startLsp () : int =
    let input  = Console.OpenStandardInput()
    let output = Console.OpenStandardOutput()

    let requestHandlings = defaultRequestHandlings ()

    let result =
        Server.start
            requestHandlings
            input
            output
            (fun (notifier, requester) -> new ConformalClient(notifier, requester))
            (fun client -> new ConformalLspServer(client))
        |> fun startFn -> startFn defaultRpc

    int result
