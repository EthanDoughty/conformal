import {
    createConnection,
    TextDocuments,
    ProposedFeatures,
    InitializeParams,
    TextDocumentSyncKind,
    Diagnostic,
    DiagnosticSeverity,
    DiagnosticTag,
    DiagnosticRelatedInformation,
    Hover,
    MarkupKind,
    CodeAction,
    CodeActionKind,
    TextEdit,
    Range,
    Position,
    DocumentSymbol,
    SymbolKind,
    Location,
} from 'vscode-languageserver/node';

import { TextDocument } from 'vscode-languageserver-textdocument';

import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// ---------------------------------------------------------------------------
// Import Fable-compiled analyzer
// ---------------------------------------------------------------------------
// eslint-disable-next-line @typescript-eslint/no-var-requires
const analyzer = require('./fable-out/Interop.js');

// Types from the Fable output (mirroring Interop.fs records)
interface SerializedDiagnostic {
    line: number;
    col: number;
    code: string;
    message: string;
    relatedLine: number | undefined;
    relatedCol: number | undefined;
}

interface FunctionSymbol {
    name: string;
    line: number;
    parms: string[];
    outputs: string[];
}

interface AnalysisResult {
    diagnostics: SerializedDiagnostic[];
    env: [string, string][];         // [varName, shapeString]
    symbols: FunctionSymbol[];
    parseError: string | undefined;
}

// ---------------------------------------------------------------------------
// Error severity codes (port of LspDiagnostics.fs ERROR_CODES)
// ---------------------------------------------------------------------------

const ERROR_CODES = new Set([
    'W_INNER_DIM_MISMATCH',
    'W_ELEMENTWISE_MISMATCH',
    'W_CONSTRAINT_CONFLICT',
    'W_HORZCAT_ROW_MISMATCH',
    'W_VERTCAT_COL_MISMATCH',
    'W_RESHAPE_MISMATCH',
    'W_INDEX_OUT_OF_BOUNDS',
    'W_DIVISION_BY_ZERO',
    'W_ARITHMETIC_TYPE_MISMATCH',
    'W_TRANSPOSE_TYPE_MISMATCH',
    'W_NEGATE_TYPE_MISMATCH',
    'W_CONCAT_TYPE_MISMATCH',
    'W_INDEX_ASSIGN_TYPE_MISMATCH',
    'W_POSSIBLY_NEGATIVE_DIM',
    'W_FUNCTION_ARG_COUNT_MISMATCH',
    'W_LAMBDA_ARG_COUNT_MISMATCH',
    'W_MULTI_ASSIGN_COUNT_MISMATCH',
    'W_MULTI_ASSIGN_NON_CALL',
    'W_MULTI_ASSIGN_BUILTIN',
    'W_PROCEDURE_IN_EXPR',
    'W_BREAK_OUTSIDE_LOOP',
    'W_CONTINUE_OUTSIDE_LOOP',
    'W_STRICT_MODE',
    'W_MLDIVIDE_DIM_MISMATCH',
    'W_MATRIX_POWER_NON_SQUARE',
]);

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------

const connection = createConnection(ProposedFeatures.all);
const documents = new TextDocuments(TextDocument);

interface CachedAnalysis {
    env: [string, string][];
    diagnostics: SerializedDiagnostic[];
    symbols: FunctionSymbol[];
    sourceHash: string;
    settingsHash: string;
}

const analysisCache = new Map<string, CachedAnalysis>();
const debounceTimers = new Map<string, NodeJS.Timeout>();

let serverSettings = {
    fixpoint: false,
    strict: false,
    analyzeOnChange: true,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function computeHash(text: string): string {
    return crypto.createHash('sha256').update(text, 'utf-8').digest('hex');
}

function settingsHash(): string {
    return computeHash(`${serverSettings.fixpoint}${serverSettings.strict}`);
}

function uriToPath(uri: string): string {
    return fileURLToPath(uri);
}

// ---------------------------------------------------------------------------
// Workspace scanning: read sibling .m files for cross-file analysis
// ---------------------------------------------------------------------------

function readExternalFiles(filePath: string): [string, string][] {
    const dir = path.dirname(filePath);
    const fileName = path.basename(filePath);
    const result: [string, string][] = [];

    try {
        if (!fs.existsSync(dir)) return result;
        const files = fs.readdirSync(dir);
        for (const f of files) {
            if (f.endsWith('.m') && f !== fileName) {
                try {
                    const content = fs.readFileSync(path.join(dir, f), 'utf-8');
                    result.push([f, content]);
                } catch { /* skip unreadable files */ }
            }
        }
    } catch { /* skip if dir listing fails */ }

    return result;
}

// ---------------------------------------------------------------------------
// Diagnostic conversion (port of LspDiagnostics.fs toLspDiagnostic)
// ---------------------------------------------------------------------------

function toLspDiagnostic(d: SerializedDiagnostic, sourceLines: string[], uri: string): Diagnostic {
    const lineNum = d.line - 1;
    const endChar = (lineNum >= 0 && lineNum < sourceLines.length)
        ? sourceLines[lineNum].length : 0;
    const startChar = d.col > 0 ? d.col - 1 : 0;

    const range: Range = {
        start: { line: lineNum, character: startChar },
        end: { line: lineNum, character: endChar },
    };

    // Severity mapping
    let severity: DiagnosticSeverity;
    if (ERROR_CODES.has(d.code)) {
        severity = DiagnosticSeverity.Error;
    } else if (d.code.startsWith('W_UNSUPPORTED_')) {
        severity = DiagnosticSeverity.Hint;
    } else {
        severity = DiagnosticSeverity.Warning;
    }

    // Tags
    const tags: DiagnosticTag[] | undefined =
        d.code.startsWith('W_UNSUPPORTED_') ? [DiagnosticTag.Unnecessary] : undefined;

    // Related information
    let relatedInformation: DiagnosticRelatedInformation[] | undefined;
    if (d.relatedLine !== undefined && d.relatedLine !== null) {
        const relLineNum = d.relatedLine - 1;
        const relEndChar = (relLineNum >= 0 && relLineNum < sourceLines.length)
            ? sourceLines[relLineNum].length : 0;
        relatedInformation = [{
            location: {
                uri,
                range: {
                    start: { line: relLineNum, character: 0 },
                    end: { line: relLineNum, character: relEndChar },
                },
            },
            message: `Related: see line ${d.relatedLine}`,
        }];
    }

    return {
        range,
        severity,
        code: d.code,
        source: 'conformal',
        message: d.message,
        tags,
        relatedInformation,
    };
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

function validate(uri: string, source: string, force = false): void {
    const srcHash = computeHash(source);
    const setHash = settingsHash();
    const sourceLines = source.split('\n');

    // Cache check
    if (!force && analysisCache.has(uri)) {
        const cached = analysisCache.get(uri)!;
        if (cached.sourceHash === srcHash && cached.settingsHash === setHash) {
            // Re-publish cached diagnostics
            const lspDiags = cached.diagnostics.map(d => toLspDiagnostic(d, sourceLines, uri));
            connection.sendDiagnostics({ uri, diagnostics: lspDiags });
            return;
        }
    }

    try {
        // Read sibling .m files for workspace awareness
        const filePath = uriToPath(uri);
        const externalFiles = readExternalFiles(filePath);

        // Run Fable-compiled analyzer
        const result: AnalysisResult = analyzer.analyzeSource(
            source,
            serverSettings.fixpoint,
            serverSettings.strict,
            externalFiles
        );

        if (result.parseError) {
            // Parse error diagnostic
            const errorDiag: Diagnostic = {
                range: { start: { line: 0, character: 0 }, end: { line: 0, character: 0 } },
                severity: DiagnosticSeverity.Error,
                source: 'conformal',
                message: `Syntax error: ${result.parseError}`,
            };
            connection.sendDiagnostics({ uri, diagnostics: [errorDiag] });
            return;
        }

        // Convert and publish diagnostics
        const lspDiags = result.diagnostics.map(d => toLspDiagnostic(d, sourceLines, uri));
        connection.sendDiagnostics({ uri, diagnostics: lspDiags });

        // Update cache
        analysisCache.set(uri, {
            env: result.env,
            diagnostics: result.diagnostics,
            symbols: result.symbols,
            sourceHash: srcHash,
            settingsHash: setHash,
        });

    } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        connection.sendDiagnostics({
            uri,
            diagnostics: [{
                range: { start: { line: 0, character: 0 }, end: { line: 0, character: 0 } },
                severity: DiagnosticSeverity.Error,
                source: 'conformal',
                message: `Internal error: ${msg}`,
            }],
        });
    }
}

// ---------------------------------------------------------------------------
// Hover (port of LspHover.fs)
// ---------------------------------------------------------------------------

function getHover(uri: string, line: number, character: number): Hover | null {
    const cached = analysisCache.get(uri);
    if (!cached) return null;

    const doc = documents.get(uri);
    if (!doc) return null;

    const lines = doc.getText().split('\n');
    if (line < 0 || line >= lines.length) return null;

    const lineText = lines[line];
    if (character < 0 || character > lineText.length) return null;

    // Extract identifier at cursor
    const identRe = /[A-Za-z_]\w*/g;
    let match: RegExpExecArray | null;
    let word = '';
    let matchStart = 0;
    let matchEnd = 0;

    while ((match = identRe.exec(lineText)) !== null) {
        const s = match.index;
        const e = s + match[0].length;
        if (s <= character && character < e) {
            word = match[0];
            matchStart = s;
            matchEnd = e;
            break;
        }
    }

    if (!word) return null;

    const hoverRange: Range = {
        start: { line, character: matchStart },
        end: { line, character: matchEnd },
    };

    // Look up in environment
    for (const [name, shape] of cached.env) {
        if (name === word) {
            return {
                contents: { kind: MarkupKind.Markdown, value: `(conformal) \`${word}\`: \`${shape}\`` },
                range: hoverRange,
            };
        }
    }

    // Check symbols (same-file functions)
    for (const sym of cached.symbols) {
        if (sym.name === word) {
            const params = sym.parms.join(', ');
            const outputs = sym.outputs.join(', ');
            return {
                contents: { kind: MarkupKind.Markdown, value: `(function) \`${word}(${params}) -> [${outputs}]\`` },
                range: hoverRange,
            };
        }
    }

    return null;
}

// ---------------------------------------------------------------------------
// Code actions (port of LspCodeActions.fs)
// ---------------------------------------------------------------------------

function getCodeActions(uri: string, diagnostics: Diagnostic[], sourceLines: string[]): CodeAction[] {
    const actions: CodeAction[] = [];

    for (const diag of diagnostics) {
        const lineNum = diag.range.start.line;
        if (lineNum < 0 || lineNum >= sourceLines.length) continue;

        const lineText = sourceLines[lineNum];
        const lineLen = lineText.length;
        const code = typeof diag.code === 'string' ? diag.code : '';

        // * -> .* (elementwise multiplication)
        if (code === 'W_INNER_DIM_MISMATCH' && diag.message.includes('elementwise multiplication')) {
            const newText = lineText.replace(/(?<!\.)\*/g, '.*');
            if (newText !== lineText) {
                actions.push({
                    title: 'Replace * with .* (elementwise)',
                    kind: CodeActionKind.QuickFix,
                    diagnostics: [diag],
                    edit: {
                        changes: {
                            [uri]: [TextEdit.replace(
                                { start: { line: lineNum, character: 0 }, end: { line: lineNum, character: lineLen } },
                                newText
                            )],
                        },
                    },
                });
            }
        }

        // && -> & and || -> |
        if (code === 'W_LOGICAL_OP_NON_SCALAR') {
            if (lineText.includes('&&')) {
                actions.push({
                    title: 'Replace && with & (elementwise)',
                    kind: CodeActionKind.QuickFix,
                    diagnostics: [diag],
                    edit: {
                        changes: {
                            [uri]: [TextEdit.replace(
                                { start: { line: lineNum, character: 0 }, end: { line: lineNum, character: lineLen } },
                                lineText.replace(/&&/g, '&')
                            )],
                        },
                    },
                });
            }
            if (lineText.includes('||')) {
                actions.push({
                    title: 'Replace || with | (elementwise)',
                    kind: CodeActionKind.QuickFix,
                    diagnostics: [diag],
                    edit: {
                        changes: {
                            [uri]: [TextEdit.replace(
                                { start: { line: lineNum, character: 0 }, end: { line: lineNum, character: lineLen } },
                                lineText.replace(/\|\|/g, '|')
                            )],
                        },
                    },
                });
            }
        }
    }

    return actions;
}

// ---------------------------------------------------------------------------
// Document symbols (port of LspSymbols.fs)
// ---------------------------------------------------------------------------

function getDocumentSymbols(uri: string): DocumentSymbol[] | null {
    const cached = analysisCache.get(uri);
    if (!cached || cached.symbols.length === 0) return null;

    const doc = documents.get(uri);
    if (!doc) return null;

    const sourceLines = doc.getText().split('\n');
    const symbols: DocumentSymbol[] = [];

    for (const sym of cached.symbols) {
        const startLine = sym.line - 1;
        const params = sym.parms.join(', ');
        const outputs = sym.outputs.join(', ');
        const detail = `(${params}) -> [${outputs}]`;

        const startLineLen = (startLine >= 0 && startLine < sourceLines.length)
            ? sourceLines[startLine].length : 0;

        symbols.push({
            name: sym.name,
            kind: SymbolKind.Function,
            range: {
                start: { line: startLine, character: 0 },
                end: { line: startLine, character: startLineLen },
            },
            selectionRange: {
                start: { line: startLine, character: 0 },
                end: { line: startLine, character: startLineLen },
            },
            detail,
        });
    }

    return symbols.length > 0 ? symbols : null;
}

// ---------------------------------------------------------------------------
// LSP event handlers
// ---------------------------------------------------------------------------

connection.onInitialize((params: InitializeParams) => {
    // Apply initialization options
    const opts = params.initializationOptions;
    if (opts && typeof opts === 'object') {
        if ('fixpoint' in opts) serverSettings.fixpoint = Boolean(opts.fixpoint);
        if ('strict' in opts) serverSettings.strict = Boolean(opts.strict);
        if ('analyzeOnChange' in opts) serverSettings.analyzeOnChange = Boolean(opts.analyzeOnChange);
    }

    return {
        capabilities: {
            textDocumentSync: TextDocumentSyncKind.Full,
            hoverProvider: true,
            codeActionProvider: {
                codeActionKinds: [CodeActionKind.QuickFix],
            },
            documentSymbolProvider: true,
        },
    };
});

// Document open: analyze immediately
documents.onDidOpen(event => {
    validate(event.document.uri, event.document.getText());
});

// Document save: analyze immediately + cross-file invalidation
documents.onDidSave(event => {
    validate(event.document.uri, event.document.getText());

    // Cross-file invalidation: re-analyze sibling files
    try {
        const savedPath = uriToPath(event.document.uri);
        const savedDir = path.dirname(savedPath);

        for (const [cachedUri] of analysisCache) {
            if (cachedUri === event.document.uri) continue;
            try {
                const cachedPath = uriToPath(cachedUri);
                if (path.dirname(cachedPath) === savedDir) {
                    const doc = documents.get(cachedUri);
                    if (doc) validate(cachedUri, doc.getText(), true);
                }
            } catch { /* skip */ }
        }
    } catch { /* skip */ }
});

// Document change: debounced analysis
documents.onDidChangeContent(event => {
    if (!serverSettings.analyzeOnChange) return;

    const uri = event.document.uri;

    // Cancel existing timer
    const existing = debounceTimers.get(uri);
    if (existing) clearTimeout(existing);

    // Schedule debounced analysis (500ms)
    const timer = setTimeout(() => {
        const doc = documents.get(uri);
        if (doc) validate(uri, doc.getText());
        debounceTimers.delete(uri);
    }, 500);

    debounceTimers.set(uri, timer);
});

// Hover
connection.onHover(params => {
    return getHover(params.textDocument.uri, params.position.line, params.position.character);
});

// Code actions
connection.onCodeAction(params => {
    const doc = documents.get(params.textDocument.uri);
    if (!doc) return null;

    const sourceLines = doc.getText().split('\n');
    const actions = getCodeActions(params.textDocument.uri, params.context.diagnostics, sourceLines);
    return actions.length > 0 ? actions : null;
});

// Document symbols
connection.onDocumentSymbol(params => {
    return getDocumentSymbols(params.textDocument.uri);
});

// Configuration changes
connection.onDidChangeConfiguration(params => {
    const settings = params?.settings;
    if (settings && typeof settings === 'object') {
        const conformal = (settings as Record<string, unknown>).conformal;
        if (conformal && typeof conformal === 'object') {
            const c = conformal as Record<string, unknown>;
            if ('fixpoint' in c) serverSettings.fixpoint = Boolean(c.fixpoint);
            if ('strict' in c) serverSettings.strict = Boolean(c.strict);
            if ('analyzeOnChange' in c) serverSettings.analyzeOnChange = Boolean(c.analyzeOnChange);
        }
    }

    // Re-analyze all cached documents with new settings
    for (const [uri] of analysisCache) {
        const doc = documents.get(uri);
        if (doc) validate(uri, doc.getText());
    }
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

documents.listen(connection);
connection.listen();
