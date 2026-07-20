// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Browser playground entry point. Wraps the Fable-compiled analyzer in a
// CodeMirror 6 editor: shape-mismatch squiggles and hover come from
// @codemirror/lint's setDiagnostics, end-of-line shape hints come from a
// small custom StateField, and both are pushed from the exact same
// analyzeSource() result in a single dispatch so they can never drift
// apart. Everything runs client-side; no network requests at runtime.

import { EditorView, basicSetup } from 'codemirror';
import { EditorState, StateEffect, StateField, Text } from '@codemirror/state';
import { Decoration, DecorationSet, WidgetType } from '@codemirror/view';
import { StreamLanguage } from '@codemirror/language';
import { octave } from '@codemirror/legacy-modes/mode/octave';
import { linter, lintGutter, setDiagnosticsEffect, Diagnostic as LintDiagnostic } from '@codemirror/lint';

import { analyzeSource } from './fable-out/Interop.js';

// ---------------------------------------------------------------------------
// Types from the Fable output (mirroring Interop.fs records; see server.ts)
// ---------------------------------------------------------------------------

interface SerializedDiagnostic {
    line: number;
    col: number;
    code: string;
    message: string;
    relatedLine: number | undefined;
    relatedCol: number | undefined;
    callStack: [string, number][];
}

interface AssignmentHint {
    name: string;
    line: number;
    col: number;
    shape: string;
}

interface SerializedParseError {
    message: string;
    startLine: number;
    startCol: number;
    endLine: number;
    endCol: number;
}

interface AnalysisResult {
    diagnostics: SerializedDiagnostic[];
    env: [string, string][];
    symbols: unknown[];
    parseError: string | undefined;
    parseErrorLine: number | undefined;
    parseErrorCol: number | undefined;
    parseErrorEndLine: number | undefined;
    parseErrorEndCol: number | undefined;
    parseErrors: SerializedParseError[];
    assignments: AssignmentHint[];
}

// ---------------------------------------------------------------------------
// Error severity codes (port of LspDiagnostics.fs ERROR_CODES, see server.ts)
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

// End the underline at the last code character on the line (port of
// server.ts's codeEndChar): drop a trailing `% comment` and whitespace.
function codeEndChar(lineText: string, startChar: number): number {
    let end = lineText.length;
    if (!lineText.includes("'")) {
        const pct = lineText.indexOf('%');
        if (pct >= 0) end = pct;
    }
    while (end > startChar + 1 && /\s/.test(lineText[end - 1])) end--;
    return end > startChar ? end : lineText.length;
}

function clampPos(doc: Text, line1: number, col1: number): number {
    const lineIdx = Math.min(Math.max(line1, 1), doc.lines);
    const line = doc.line(lineIdx);
    const ch = Math.min(Math.max(col1 - 1, 0), line.length);
    return line.from + ch;
}

function diagToLint(d: SerializedDiagnostic, doc: Text): LintDiagnostic | null {
    if (d.line < 1 || d.line > doc.lines) return null;
    const line = doc.line(d.line);
    const startChar = d.col > 0 ? d.col - 1 : 0;
    const endChar = codeEndChar(line.text, startChar);
    const from = Math.min(line.to, line.from + startChar);
    const to = Math.max(from + 1, Math.min(line.to, line.from + endChar));
    return {
        from, to,
        severity: ERROR_CODES.has(d.code) ? 'error' : 'warning',
        source: 'conformal',
        message: `${d.code}: ${d.message}`,
    };
}

function parseErrorToLint(message: string, startLine: number, startCol: number, endLine: number, endCol: number, doc: Text): LintDiagnostic {
    const from = clampPos(doc, startLine, startCol);
    const to = Math.max(from + 1, Math.min(doc.length, clampPos(doc, endLine, endCol)));
    return { from, to, severity: 'error', source: 'conformal', message: `Syntax error: ${message}` };
}

// ---------------------------------------------------------------------------
// Shape hint widget (end-of-line ": matrix[3 x 4]" decoration)
// ---------------------------------------------------------------------------

class ShapeHintWidget extends WidgetType {
    constructor(readonly text: string) { super(); }
    eq(other: WidgetType): boolean { return other instanceof ShapeHintWidget && other.text === this.text; }
    toDOM(): HTMLElement {
        const span = document.createElement('span');
        span.className = 'pg-shape-hint';
        span.textContent = this.text;
        return span;
    }
    ignoreEvent(): boolean { return true; }
}

// ---------------------------------------------------------------------------
// Single StateEffect/StateField pair: the latest analyzeSource() result.
// Shape hints are decorations computed straight from this field, and the
// same result also drives the @codemirror/lint diagnostics in the same
// dispatch, so hints and squiggles are always in sync.
// ---------------------------------------------------------------------------

const setAnalysisEffect = StateEffect.define<AssignmentHint[]>();

const analysisField = StateField.define<AssignmentHint[]>({
    create: () => [],
    update(value, tr) {
        for (const e of tr.effects) {
            if (e.is(setAnalysisEffect)) value = e.value;
        }
        return value;
    },
});

function buildHintDecorations(state: EditorState): DecorationSet {
    const assignments = state.field(analysisField);
    const doc = state.doc;
    const decos = [];
    for (const a of assignments) {
        if (a.line < 1 || a.line > doc.lines) continue;
        const line = doc.line(a.line);
        const pos = Math.min(line.to, line.from + (a.col - 1) + a.name.length);
        decos.push(Decoration.widget({ widget: new ShapeHintWidget(`: ${a.shape}`), side: 1 }).range(pos));
    }
    return Decoration.set(decos, true);
}

const hintDecorations = EditorView.decorations.compute([analysisField], buildHintDecorations);

// ---------------------------------------------------------------------------
// Example snippets
// ---------------------------------------------------------------------------

const EXAMPLES: { label: string; code: string }[] = [
    {
        label: 'Dimension mismatch',
        code: 'A = zeros(3,4);\nB = ones(5,6);\nC = A * B;\n',
    },
    {
        label: 'Clean shapes',
        code: 'A = zeros(3, 4);\nB = ones(4, 5);\nC = A * B;\nD = C\';\n',
    },
    {
        label: 'Strict-only warning',
        code: "A = zeros(3, 3);\nr = A + 'error';\n",
    },
];

// ---------------------------------------------------------------------------
// Diagnostics panel (below the editor)
// ---------------------------------------------------------------------------

function renderPanel(panel: HTMLElement, view: EditorView, result: AnalysisResult): void {
    panel.textContent = '';

    const doc = view.state.doc;
    const rows: { line: number; col: number; code: string; message: string; from: number; to: number }[] = [];

    if (result.parseError) {
        const line = result.parseErrorLine ?? 1;
        const col = result.parseErrorCol ?? 1;
        const endLine = result.parseErrorEndLine ?? line;
        const endCol = result.parseErrorEndCol ?? col + 1;
        const from = clampPos(doc, line, col);
        const to = Math.max(from + 1, Math.min(doc.length, clampPos(doc, endLine, endCol)));
        rows.push({ line, col, code: 'SYNTAX_ERROR', message: result.parseError, from, to });
    }

    for (const pe of result.parseErrors) {
        const from = clampPos(doc, pe.startLine, pe.startCol);
        const to = Math.max(from + 1, Math.min(doc.length, clampPos(doc, pe.endLine, pe.endCol)));
        rows.push({ line: pe.startLine, col: pe.startCol, code: 'SYNTAX_ERROR', message: pe.message, from, to });
    }

    for (const d of result.diagnostics) {
        const lint = diagToLint(d, doc);
        rows.push({
            line: d.line, col: d.col, code: d.code, message: d.message,
            from: lint ? lint.from : 0, to: lint ? lint.to : 0,
        });
    }

    if (rows.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'pg-panel-empty';
        empty.textContent = 'No diagnostics.';
        panel.appendChild(empty);
        return;
    }

    for (const row of rows) {
        const el = document.createElement('div');
        el.className = 'pg-panel-row';
        el.tabIndex = 0;

        const loc = document.createElement('span');
        loc.className = 'pg-panel-loc';
        loc.textContent = `${row.line}:${row.col}`;

        const code = document.createElement('span');
        code.className = 'pg-panel-code';
        code.textContent = row.code;

        const msg = document.createElement('span');
        msg.className = 'pg-panel-msg';
        msg.textContent = row.message;

        el.appendChild(loc);
        el.appendChild(code);
        el.appendChild(msg);

        const jump = () => {
            view.dispatch({ selection: { anchor: row.from, head: row.to }, scrollIntoView: true });
            view.focus();
        };
        el.addEventListener('click', jump);
        el.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); jump(); }
        });

        panel.appendChild(el);
    }
}

// ---------------------------------------------------------------------------
// Analysis pipeline
// ---------------------------------------------------------------------------

function runAnalysis(view: EditorView, panel: HTMLElement, strict: boolean): void {
    const source = view.state.doc.toString();
    let result: AnalysisResult;
    try {
        result = analyzeSource(source, true, strict, []);
    } catch (e) {
        // analyzeSource is expected to report parse errors rather than throw,
        // but the page must never crash regardless of input, so guard anyway.
        result = {
            diagnostics: [], env: [], symbols: [],
            parseError: e instanceof Error ? e.message : String(e),
            parseErrorLine: 1, parseErrorCol: 1, parseErrorEndLine: 1, parseErrorEndCol: 1,
            parseErrors: [], assignments: [],
        };
    }

    const doc = view.state.doc;
    const lintDiags: LintDiagnostic[] = [];

    if (result.parseError) {
        lintDiags.push(parseErrorToLint(
            result.parseError,
            result.parseErrorLine ?? 1, result.parseErrorCol ?? 1,
            result.parseErrorEndLine ?? (result.parseErrorLine ?? 1),
            result.parseErrorEndCol ?? (result.parseErrorCol ?? 1) + 1,
            doc,
        ));
    }
    for (const pe of result.parseErrors) {
        lintDiags.push(parseErrorToLint(pe.message, pe.startLine, pe.startCol, pe.endLine, pe.endCol, doc));
    }
    for (const d of result.diagnostics) {
        const lint = diagToLint(d, doc);
        if (lint) lintDiags.push(lint);
    }

    // Single dispatch: shape hints (custom effect) and squiggles/hover
    // (lint's own setDiagnosticsEffect) both come from this one result.
    view.dispatch({
        effects: [
            setAnalysisEffect.of(result.assignments || []),
            setDiagnosticsEffect.of(lintDiags),
        ],
    });

    renderPanel(panel, view, result);
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

function main(): void {
    const editorHost = document.getElementById('pg-editor');
    const panel = document.getElementById('pg-panel');
    const strictBox = document.getElementById('pg-strict') as HTMLInputElement | null;
    const exampleSelect = document.getElementById('pg-examples') as HTMLSelectElement | null;

    if (!editorHost || !panel || !strictBox || !exampleSelect) return;

    exampleSelect.textContent = '';
    for (let i = 0; i < EXAMPLES.length; i++) {
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = EXAMPLES[i].label;
        exampleSelect.appendChild(opt);
    }

    let debounceTimer: number | undefined;
    const scheduleAnalysis = (view: EditorView) => {
        if (debounceTimer !== undefined) window.clearTimeout(debounceTimer);
        debounceTimer = window.setTimeout(() => {
            debounceTimer = undefined;
            runAnalysis(view, panel, strictBox.checked);
        }, 300);
    };

    const state = EditorState.create({
        doc: EXAMPLES[0].code,
        extensions: [
            basicSetup,
            StreamLanguage.define(octave),
            analysisField,
            hintDecorations,
            linter(null, { delay: 300 }),
            lintGutter(),
            EditorView.theme({
                '&': { height: '420px', fontSize: '14px' },
                '.cm-scroller': { overflow: 'auto', fontFamily: "'JetBrains Mono', 'Fira Code', monospace" },
            }),
            EditorView.updateListener.of((update) => {
                if (update.docChanged) scheduleAnalysis(update.view);
            }),
        ],
    });

    const view = new EditorView({ state, parent: editorHost });

    strictBox.addEventListener('change', () => runAnalysis(view, panel, strictBox.checked));

    exampleSelect.addEventListener('change', () => {
        const idx = Number(exampleSelect.value);
        const example = EXAMPLES[idx];
        if (!example) return;
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: example.code } });
        runAnalysis(view, panel, strictBox.checked);
    });

    // Analyze once on load.
    runAnalysis(view, panel, strictBox.checked);
}

main();
