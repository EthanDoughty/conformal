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
import { StreamLanguage, HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { octave } from '@codemirror/legacy-modes/mode/octave';
import { linter, lintGutter, setDiagnosticsEffect, Diagnostic as LintDiagnostic } from '@codemirror/lint';
import { tags } from '@lezer/highlight';

// @ts-expect-error -- Fable emits no declarations; the interfaces below mirror Interop.fs
import { analyzeSource } from './fable-out/Interop.js';

import { EXAMPLE_GROUPS, EXAMPLES, Example, MatrixMeta, generateCode, defaultParamValues } from './playground-examples';

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
    'W_MRDIVIDE_DIM_MISMATCH',
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
        // Anchor after the variable name as it appears in the line text; the
        // IR column is not the variable position for for-loop statements.
        // MATLAB identifiers contain no regex metacharacters.
        const match = new RegExp(`\\b${a.name}\\b`).exec(line.text);
        const anchor = match ? match.index + a.name.length : (a.col - 1) + a.name.length;
        const pos = Math.min(line.to, line.from + anchor);
        decos.push(Decoration.widget({ widget: new ShapeHintWidget(`: ${a.shape}`), side: 1 }).range(pos));
    }
    return Decoration.set(decos, true);
}

const hintDecorations = EditorView.decorations.compute([analysisField], buildHintDecorations);

// ---------------------------------------------------------------------------
// Example snippets
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Dark theme matched to the site palette. Without this, basicSetup's default
// highlight style (designed for light backgrounds) is illegible on the dark
// page, and the lint tooltip stays light while inheriting the page's light
// text. Severity is encoded in lightness and marker shape, not red-vs-green
// hue, so it survives color vision deficiency. rgb() in the inline SVGs
// avoids '#', which would terminate a data: URI.
// ---------------------------------------------------------------------------

const ERROR_COLOR = 'rgb(255,110,95)';
const WARN_COLOR = 'rgb(227,179,65)';

function squiggle(rgb: string): string {
    return `url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="6" height="3"><path d="m0 3 l2 -2 l1 0 l2 2 l1 0" stroke="${rgb}" fill="none" stroke-width=".7"/></svg>')`;
}

// Error marker is a circle, warning a triangle: shape carries the severity
// alongside color.
const ERROR_MARKER = `url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="40" height="40"><circle cx="20" cy="20" r="14" fill="${ERROR_COLOR}"/></svg>')`;
const WARN_MARKER = `url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="40" height="40"><path d="M20 6 L37 34 L3 34 Z" fill="${WARN_COLOR}"/></svg>')`;

const conformalTheme = EditorView.theme({
    '&': { height: '420px', fontSize: '14px', backgroundColor: '#18160f', color: '#e8e4dc' },
    '.cm-scroller': { overflow: 'auto', fontFamily: "'JetBrains Mono', 'Fira Code', monospace" },
    '.cm-content': { caretColor: '#d4a021' },
    '.cm-cursor, .cm-dropCursor': { borderLeftColor: '#d4a021' },
    '&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground, ::selection':
        { backgroundColor: '#3a3426' },
    '.cm-activeLine': { backgroundColor: 'rgba(232,228,220,0.05)' },
    '.cm-gutters': { backgroundColor: '#18160f', color: '#8a8478', border: 'none', borderRight: '1px solid #302d26' },
    '.cm-activeLineGutter': { backgroundColor: '#1e1c18', color: '#e8e4dc' },
    '.cm-tooltip': { backgroundColor: '#262218', color: '#e8e4dc', border: '1px solid #3a362c' },
    '.cm-tooltip .cm-tooltip-arrow:before': { borderTopColor: '#3a362c', borderBottomColor: '#3a362c' },
    '.cm-tooltip .cm-tooltip-arrow:after': { borderTopColor: '#262218', borderBottomColor: '#262218' },
    '.cm-diagnostic-error': { borderLeft: `5px solid ${ERROR_COLOR}` },
    '.cm-diagnostic-warning': { borderLeft: `5px solid ${WARN_COLOR}` },
    '.cm-lintRange-error': { backgroundImage: squiggle(ERROR_COLOR) },
    '.cm-lintRange-warning': { backgroundImage: squiggle(WARN_COLOR) },
    '.cm-lintPoint-error:after': { borderBottomColor: ERROR_COLOR },
    '.cm-lintPoint-warning:after': { borderBottomColor: WARN_COLOR },
    '.cm-lint-marker-error': { content: ERROR_MARKER },
    '.cm-lint-marker-warning': { content: WARN_MARKER },
}, { dark: true });

// All foregrounds hold WCAG AA (4.5:1) or better on #18160f.
const conformalHighlight = HighlightStyle.define([
    { tag: tags.keyword, color: '#e3b341' },
    { tag: [tags.atom, tags.bool], color: '#d4a021' },
    { tag: tags.comment, color: '#9a9488', fontStyle: 'italic' },
    { tag: tags.string, color: '#d69a6a' },
    { tag: tags.number, color: '#8fb7dd' },
    { tag: [tags.operator, tags.punctuation, tags.bracket], color: '#c9c3b6' },
    { tag: tags.variableName, color: '#e8e4dc' },
    { tag: tags.special(tags.variableName), color: '#8fb7dd' },
    { tag: tags.meta, color: '#9a9488' },
]);

// ---------------------------------------------------------------------------
// Analysis options, read from the toolbar on every run.
// ---------------------------------------------------------------------------

interface PlaygroundOptions {
    strict: boolean;
    fixpoint: boolean;
    annotations: boolean;
}

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

function runAnalysis(view: EditorView, panel: HTMLElement, opts: PlaygroundOptions): void {
    const source = view.state.doc.toString();
    let result: AnalysisResult;
    try {
        result = analyzeSource(source, opts.fixpoint, opts.strict, []);
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
            setAnalysisEffect.of(opts.annotations ? (result.assignments || []) : []),
            setDiagnosticsEffect.of(lintDiags),
        ],
    });

    renderPanel(panel, view, result);
}

// ---------------------------------------------------------------------------
// Literal-matrix params: a value like "[4, 1, 0; 1, 3, 1; 0, 1, 2]" is shown
// as a small grid of number squares instead of one text box. Only clean,
// short, small literal matrices qualify; constructors (zeros(100,3)), symbolic
// expressions, scalars, and long values stay as a plain text input.
// ---------------------------------------------------------------------------

interface LiteralMatrix { rows: number; cols: number; cells: string[][]; }

const MATRIX_ENTRY = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

function parseLiteralMatrix(value: string): LiteralMatrix | null {
    const v = (value || '').trim();
    if (v.length < 3 || v[0] !== '[' || v[v.length - 1] !== ']') return null;
    const inner = v.slice(1, -1).trim();
    if (inner === '') return null;
    const cells: string[][] = [];
    let cols = -1;
    for (const rowText of inner.split(';')) {
        const parts = rowText.trim().split(',').map(s => s.trim());
        if (cols === -1) cols = parts.length;
        else if (parts.length !== cols) return null;   // ragged: not a clean grid
        for (const p of parts) if (!MATRIX_ENTRY.test(p)) return null; // any non-numeric entry
        cells.push(parts);
    }
    return { rows: cells.length, cols, cells };
}

function matrixEligible(m: LiteralMatrix | null): m is LiteralMatrix {
    if (!m) return false;
    if (m.rows === 1 && m.cols === 1) return false;   // a 1x1 is a scalar, keep text
    if (m.rows > 4 || m.cols > 4 || m.rows * m.cols > 16) return false;
    for (const r of m.cells) for (const c of r) if (c.length > 6) return false; // long values read poorly
    return true;
}

function matrixToString(cells: string[][]): string {
    return '[' + cells.map(r => r.join(', ')).join('; ') + ']';
}

const MATRIX_MAX = 6;   // resize cap per dimension

// Build the square-cell matrix knob into `block`: an editable grid of number
// squares with optional row/column teaching labels and resize steppers. Labels
// show only at the original shape (resizing would misalign them). Every edit or
// resize reassembles the "[a, b; c, d]" string and calls onChange.
function buildMatrixKnob(
    block: HTMLElement, key: string, initial: LiteralMatrix,
    meta: MatrixMeta | undefined, onChange: (value: string) => void,
): void {
    const origRows = initial.rows, origCols = initial.cols;
    let cells: string[][] = initial.cells.map(r => r.slice());
    // A matrix with three or more columns, or one carrying labels, is too wide
    // for a single grid track, so it takes the full panel width.
    if (origCols >= 3 || meta) block.classList.add('pg-param-wide');

    const wrap = document.createElement('div');
    wrap.className = 'pg-matrix-wrap';

    const span = (cls: string, text?: string): HTMLElement => {
        const s = document.createElement('span');
        s.className = cls;
        if (text != null) s.textContent = text;
        return s;
    };
    const commit = () => onChange(matrixToString(cells));

    const resize = (dim: 'row' | 'col', delta: number) => {
        const rows = cells.length, cols = cells[0].length;
        if (dim === 'row') {
            if (delta > 0 && rows < MATRIX_MAX) cells.push(new Array(cols).fill('0'));
            else if (delta < 0 && rows > 1 && rows * cols > 2) cells.pop();
        } else {
            if (delta > 0 && cols < MATRIX_MAX) cells.forEach(r => r.push('0'));
            else if (delta < 0 && cols > 1 && rows * cols > 2) cells.forEach(r => r.pop());
        }
        commit();
        render();
    };

    const stepper = (labelText: string, dim: 'row' | 'col', count: number, rows: number, cols: number): HTMLElement => {
        const grp = span('pg-matrix-step');
        grp.appendChild(span('pg-matrix-steplabel', `${labelText} ${count}`));
        const minus = document.createElement('button');
        minus.type = 'button'; minus.className = 'pg-matrix-btn'; minus.textContent = '−';
        minus.disabled = (dim === 'row' ? rows : cols) <= 1 || rows * cols <= 2;
        minus.addEventListener('click', () => resize(dim, -1));
        const plus = document.createElement('button');
        plus.type = 'button'; plus.className = 'pg-matrix-btn'; plus.textContent = '+';
        plus.disabled = (dim === 'row' ? rows : cols) >= MATRIX_MAX;
        plus.addEventListener('click', () => resize(dim, +1));
        grp.appendChild(minus);
        grp.appendChild(plus);
        return grp;
    };

    const render = () => {
        wrap.textContent = '';
        const rows = cells.length, cols = cells[0].length;
        const atOriginal = rows === origRows && cols === origCols;
        const colL = atOriginal && meta?.cols && meta.cols.length === cols ? meta.cols : null;
        const rowL = atOriginal && meta?.rows && meta.rows.length === rows ? meta.rows : null;
        const rowD = atOriginal && meta?.rowDesc && meta.rowDesc.length === rows ? meta.rowDesc : null;

        const table = span('pg-matrix');
        if (colL) {
            const hr = span('pg-matrix-row');
            if (rowL) hr.appendChild(span('pg-matrix-corner'));
            for (let j = 0; j < cols; j++) hr.appendChild(span('pg-matrix-collabel', colL[j]));
            table.appendChild(hr);
        }
        for (let i = 0; i < rows; i++) {
            const rr = span('pg-matrix-row');
            if (rowL) rr.appendChild(span('pg-matrix-rowlabel', rowL[i]));
            for (let j = 0; j < cols; j++) {
                const ci = document.createElement('input');
                ci.type = 'text';
                ci.spellcheck = false;
                ci.className = 'pg-matrix-cell';
                ci.value = cells[i][j];
                const ii = i, jj = j;
                ci.addEventListener('input', () => { cells[ii][jj] = ci.value.trim() || '0'; commit(); });
                rr.appendChild(ci);
            }
            if (rowD) rr.appendChild(span('pg-matrix-rowdesc', rowD[i]));
            table.appendChild(rr);
        }
        wrap.appendChild(table);

        const controls = span('pg-matrix-controls');
        if (rows === 1) controls.appendChild(stepper('length', 'col', cols, rows, cols));
        else if (cols === 1) controls.appendChild(stepper('length', 'row', rows, rows, cols));
        else {
            controls.appendChild(stepper('rows', 'row', rows, rows, cols));
            controls.appendChild(stepper('cols', 'col', cols, rows, cols));
        }
        wrap.appendChild(controls);
    };

    render();
    block.appendChild(wrap);
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

function main(): void {
    const editorHost = document.getElementById('pg-editor');
    const panel = document.getElementById('pg-panel');
    const strictBox = document.getElementById('pg-strict') as HTMLInputElement | null;
    const fixpointBox = document.getElementById('pg-fixpoint') as HTMLInputElement | null;
    const annotationsBox = document.getElementById('pg-annotations') as HTMLInputElement | null;
    const liveBox = document.getElementById('pg-live') as HTMLInputElement | null;
    const commentsBox = document.getElementById('pg-comments') as HTMLInputElement | null;
    const analyzeBtn = document.getElementById('pg-analyze') as HTMLButtonElement | null;
    const exampleSelect = document.getElementById('pg-examples') as HTMLSelectElement | null;
    const paramsHost = document.getElementById('pg-params');

    if (!editorHost || !panel || !strictBox || !fixpointBox || !annotationsBox
        || !liveBox || !commentsBox || !analyzeBtn || !exampleSelect) return;

    const options = (): PlaygroundOptions => ({
        strict: strictBox.checked,
        fixpoint: fixpointBox.checked,
        annotations: annotationsBox.checked,
    });

    const noteEl = document.getElementById('pg-example-note');
    const searchInput = document.getElementById('pg-search') as HTMLInputElement | null;
    const categorySelect = document.getElementById('pg-category') as HTMLSelectElement | null;

    if (categorySelect) {
        const all = document.createElement('option');
        all.value = '';
        all.textContent = `All categories (${EXAMPLES.length})`;
        categorySelect.appendChild(all);
        for (const g of EXAMPLE_GROUPS) {
            const opt = document.createElement('option');
            opt.value = g.group;
            opt.textContent = `${g.group} (${g.items.length})`;
            categorySelect.appendChild(opt);
        }
    }

    // Rebuild the example list from the search box and category filter.
    // Option values stay flat indices into EXAMPLES, so filtering never
    // changes which example a value refers to. Rebuilding does not load an
    // example into the editor; only a user change on the select does.
    const rebuildExampleList = () => {
        const q = searchInput ? searchInput.value.trim().toLowerCase() : '';
        const cat = categorySelect ? categorySelect.value : '';
        const prev = exampleSelect.value;
        exampleSelect.textContent = '';
        let flatIndex = 0;
        let shown = 0;
        for (const g of EXAMPLE_GROUPS) {
            const inCategory = cat === '' || g.group === cat;
            const og = document.createElement('optgroup');
            for (const item of g.items) {
                const idx = flatIndex++;
                if (!inCategory) continue;
                if (q !== ''
                    && !item.label.toLowerCase().includes(q)
                    && !(item.note || '').toLowerCase().includes(q)) continue;
                const opt = document.createElement('option');
                opt.value = String(idx);
                opt.textContent = item.label;
                og.appendChild(opt);
            }
            if (og.childElementCount > 0) {
                // Count reflects what is actually in the group, so it equals the
                // category total when unfiltered and the match count while searching.
                og.label = `${g.group} (${og.childElementCount})`;
                exampleSelect.appendChild(og);
                shown += og.childElementCount;
            }
        }
        if (shown === 0) {
            const none = document.createElement('option');
            none.value = '';
            none.disabled = true;
            none.textContent = 'No matching examples';
            exampleSelect.appendChild(none);
        } else if (prev !== '' && exampleSelect.querySelector(`option[value="${prev}"]`)) {
            exampleSelect.value = prev;
        }
    };
    rebuildExampleList();

    if (searchInput) searchInput.addEventListener('input', rebuildExampleList);
    if (categorySelect) categorySelect.addEventListener('change', rebuildExampleList);

    // While variable comments are on, the note lives inside the generated
    // code as its header line, so the caption would just repeat it.
    const showNote = (example: Example | undefined) => {
        if (!noteEl) return;
        const note = example && example.note ? example.note : '';
        noteEl.textContent = note;
        noteEl.hidden = note === '' || commentsBox.checked;
    };
    showNote(EXAMPLES[0]);

    let debounceTimer: number | undefined;
    const scheduleAnalysis = (view: EditorView) => {
        if (debounceTimer !== undefined) window.clearTimeout(debounceTimer);
        debounceTimer = window.setTimeout(() => {
            debounceTimer = undefined;
            runAnalysis(view, panel, options());
        }, 300);
    };

    // Template state. `dirty` flips when the user edits the document by
    // hand; from then on the param panel hides and the generator stays away
    // from the editor until an example is selected again, so pasted code is
    // never overwritten. `programmatic` marks our own dispatches (CodeMirror
    // runs update listeners synchronously inside dispatch).
    let current: Example = EXAMPLES[0];
    let paramValues = defaultParamValues(current);
    let dirty = false;
    let programmatic = false;

    const state = EditorState.create({
        doc: generateCode(current, paramValues, commentsBox.checked),
        extensions: [
            basicSetup,
            StreamLanguage.define(octave),
            analysisField,
            hintDecorations,
            linter(null, { delay: 300 }),
            lintGutter(),
            conformalTheme,
            syntaxHighlighting(conformalHighlight),
            EditorView.updateListener.of((update) => {
                if (update.docChanged && !programmatic) {
                    dirty = true;
                    if (paramsHost) paramsHost.hidden = true;
                    if (liveBox.checked) scheduleAnalysis(update.view);
                }
            }),
        ],
    });

    const view = new EditorView({ state, parent: editorHost });

    const setDoc = (code: string) => {
        programmatic = true;
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: code } });
        programmatic = false;
    };

    const regenerate = () => {
        if (dirty) return;
        setDoc(generateCode(current, paramValues, commentsBox.checked));
        runAnalysis(view, panel, options());
    };

    // Param edits regenerate on a short debounce so half-typed values do
    // not flash parse errors while the user is still typing.
    let paramTimer: number | undefined;
    const scheduleRegenerate = () => {
        if (paramTimer !== undefined) window.clearTimeout(paramTimer);
        paramTimer = window.setTimeout(() => {
            paramTimer = undefined;
            regenerate();
        }, 350);
    };

    // Each entry shows the short name, the input, and the variable's docs
    // description underneath, so the knob explains itself without hovering.
    const buildParamsPanel = () => {
        if (!paramsHost) return;
        paramsHost.textContent = '';
        const params = current.params || [];
        if (params.length === 0 || dirty) {
            paramsHost.hidden = true;
            return;
        }
        // Header row: the panel title on the left, a reset control on the right.
        const head = document.createElement('div');
        head.className = 'pg-params-head';
        const title = document.createElement('span');
        title.className = 'pg-params-title';
        title.textContent = 'Template values';
        head.appendChild(title);
        const reset = document.createElement('button');
        reset.type = 'button';
        reset.className = 'pg-reset';
        reset.textContent = 'Reset';
        reset.title = 'Restore every value to the template default';
        reset.addEventListener('click', () => {
            paramValues = defaultParamValues(current);
            buildParamsPanel();
            regenerate();
        });
        head.appendChild(reset);
        paramsHost.appendChild(head);

        // Each knob is a grid cell: name on top, input, then its description.
        // The grid aligns the cells regardless of description length.
        const grid = document.createElement('div');
        grid.className = 'pg-params-grid';
        for (const p of params) {
            const block = document.createElement('label');
            block.className = 'pg-param';
            const name = document.createElement('span');
            name.className = 'pg-param-name';
            name.textContent = p.label;
            block.appendChild(name);

            const desc = (current.docs || {})[p.key] || '';
            const mat = parseLiteralMatrix(paramValues[p.key] ?? '');
            if (matrixEligible(mat)) {
                buildMatrixKnob(block, p.key, mat, (current.matrixMeta || {})[p.key],
                    (v) => { paramValues[p.key] = v; scheduleRegenerate(); });
            } else {
                const input = document.createElement('input');
                input.type = 'text';
                input.spellcheck = false;
                input.value = paramValues[p.key] ?? '';
                input.addEventListener('input', () => {
                    paramValues[p.key] = input.value;
                    scheduleRegenerate();
                });
                if (desc !== '') input.title = desc;
                block.appendChild(input);
            }

            if (desc !== '') {
                const d = document.createElement('span');
                d.className = 'pg-param-desc';
                d.textContent = desc;
                block.appendChild(d);
            }
            grid.appendChild(block);
        }
        paramsHost.appendChild(grid);
        paramsHost.hidden = false;
    };

    const loadExample = (example: Example) => {
        current = example;
        dirty = false;
        paramValues = defaultParamValues(example);
        showNote(example);
        buildParamsPanel();
        setDoc(generateCode(example, paramValues, commentsBox.checked));
        runAnalysis(view, panel, options());
    };

    for (const box of [strictBox, fixpointBox, annotationsBox]) {
        box.addEventListener('change', () => runAnalysis(view, panel, options()));
    }

    commentsBox.addEventListener('change', () => {
        showNote(current);
        regenerate();
    });

    // The Analyze button is only visible while analyze-as-you-type is off;
    // re-enabling live analysis catches up on edits made while it was off.
    analyzeBtn.hidden = liveBox.checked;
    liveBox.addEventListener('change', () => {
        analyzeBtn.hidden = liveBox.checked;
        if (liveBox.checked) runAnalysis(view, panel, options());
    });
    analyzeBtn.addEventListener('click', () => runAnalysis(view, panel, options()));

    exampleSelect.addEventListener('change', () => {
        const idx = Number(exampleSelect.value);
        const example = EXAMPLES[idx];
        if (!example) return;
        loadExample(example);
    });

    // Analyze once on load.
    buildParamsPanel();
    runAnalysis(view, panel, options());
}

main();
