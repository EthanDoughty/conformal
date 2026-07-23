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

interface Example {
    label: string;
    code: string;
    note?: string;   // shown as a muted caption while the example is selected
}

// Every example is verified against the analyzer before shipping: the error
// ones emit exactly the named warning, the clean ones emit nothing.
const EXAMPLE_GROUPS: { group: string; items: Example[] }[] = [
    {
        group: 'Dimension errors',
        items: [
            {
                label: 'Matrix multiply',
                code: 'A = zeros(3,4);\nB = ones(5,6);\nC = A * B;\n',
            },
            {
                label: 'Elementwise op',
                code: 'A = zeros(2, 3);\nB = ones(3, 2);\nC = A .* B;\n',
            },
            {
                label: 'Concatenation',
                code: 'top = zeros(2, 3);\nbottom = ones(2, 4);\nM = [top; bottom];\n',
            },
            {
                label: 'Indexing',
                code: 'A = zeros(2, 2);\nb = A(3, 1);\n',
            },
        ],
    },
    {
        group: 'Automotive',
        items: [
            {
                label: 'ADAS sensor fusion',
                code: 'x_cam = [12.1; 3.4];\nP_cam = [0.6, 0; 0, 0.9];\nx_rad = [12.4; 3.1];\nP_rad = [0.2, 0; 0, 1.5];\nW1 = inv(P_cam);\nW2 = inv(P_rad);\nP_f = inv(W1 + W2);\nx_f = P_f * (W1 * x_cam + W2 * x_rad);\n',
                note: 'Camera and radar estimates merged by their covariances.',
            },
            {
                label: 'Battery RC model',
                code: 'dt = 1;\nCq = 8000;\nRrc = 0.015;\nCrc = 2400;\nA = [1, 0; 0, 1 - dt / (Rrc * Crc)];\nB = [-dt / Cq; dt / Crc];\nx = [0.9; 0];\nfor k = 1:60\n    x = A * x + B * 12;\nend\nv = 3.6 + 0.7 * x(1) - x(2);\n',
                note: 'An equivalent-circuit battery state stepped over a discharge.',
            },
            {
                label: 'Wheel slip ratios',
                code: 'wheel = [21.8, 22.1, 20.4, 21.9];\nrw = 0.32;\nvx = 7.1;\nslip = (vx - rw * wheel) / vx;\nworst = max(slip);\n',
                note: 'Elementwise slip computation across all four wheels.',
            },
        ],
    },
    {
        group: 'Aviation & navigation',
        items: [
            {
                label: '3D rigid transform',
                code: 'theta = pi / 4;\nR = [cos(theta), -sin(theta), 0; sin(theta), cos(theta), 0; 0, 0, 1];\nt = [1; 2; 0];\nP = zeros(3, 25);\nmoved = R * P + t * ones(1, 25);\n',
                note: 'Rotate and translate a point cloud in one expression.',
            },
            {
                label: 'ECEF to NED transform',
                code: 'lat = 0.68;\nlon = -1.66;\nR = [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat); -sin(lon), cos(lon), 0; -cos(lat) * cos(lon), -cos(lat) * sin(lon), -sin(lat)];\ndp = [1200; -340; 560];\nned = R * dp;\n',
                note: 'The local navigation frame built from latitude and longitude.',
            },
            {
                label: 'Euler DCM chain',
                code: "phi = 0.1;\ntheta = 0.05;\npsi = 1.2;\nRx = [1, 0, 0; 0, cos(phi), sin(phi); 0, -sin(phi), cos(phi)];\nRy = [cos(theta), 0, -sin(theta); 0, 1, 0; sin(theta), 0, cos(theta)];\nRz = [cos(psi), sin(psi), 0; -sin(psi), cos(psi), 0; 0, 0, 1];\nC = Rx * Ry * Rz;\nv_ned = C' * [25; 2; -1];\n",
                note: 'A 3-2-1 rotation sequence assembled and applied to a body vector.',
            },
            {
                label: 'GPS trilateration',
                code: "sats = [15600, 7540, 20140; 18760, 2750, 18610; 17610, 14630, 13480; 19170, 610, 18390];\nrho = [21000; 21500; 22000; 21500];\nx = [0; 0; 0];\nfor it = 1:5\n    d = sats - ones(4, 1) * x';\n    r = sqrt(d(:, 1).^2 + d(:, 2).^2 + d(:, 3).^2);\n    G = -d ./ (r * ones(1, 3));\n    dx = inv(G' * G) * G' * (rho - r);\n    x = x + dx;\nend\n",
                note: 'Five Gauss-Newton steps from pseudoranges to a position fix.',
            },
        ],
    },
    {
        group: 'Controls & estimation',
        items: [
            {
                label: 'Kalman filter update',
                code: "x = zeros(4, 1);\nP = eye(4);\nH = zeros(2, 4);\nR = eye(2);\nz = zeros(2, 1);\nS = H * P * H' + R;\nK = P * H' * inv(S);\nx = x + K * (z - H * x);\nP = (eye(4) - K * H) * P;\n",
                note: 'A real filter update, tracked shape by shape with nothing flagged.',
            },
            {
                label: 'State-space simulation',
                code: 'A = eye(4);\nB = zeros(4, 2);\nC = zeros(2, 4);\nx = zeros(4, 1);\nu = ones(2, 1);\nfor k = 1:10\n    x = A * x + B * u;\nend\ny = C * x;\n',
                note: 'Ten simulation steps in a loop, every shape stable throughout.',
            },
        ],
    },
    {
        group: 'Data & machine learning',
        items: [
            {
                label: 'Covariance matrix',
                code: "X = zeros(200, 3);\nmu = mean(X);\nXc = X - ones(200, 1) * mu;\nC = (Xc' * Xc) / (200 - 1);\n",
                note: 'Center the data, then form the covariance matrix.',
            },
            {
                label: 'Least squares fit',
                code: "X = zeros(100, 3);\ny = zeros(100, 1);\nXtX = X' * X;\nbeta = inv(XtX) * X' * y;\nr = y - X * beta;\nsse = r' * r;\n",
                note: 'Normal equations for a linear fit, from data matrix to residual.',
            },
            {
                label: 'Neural net forward pass',
                code: 'x = zeros(8, 1);\nW1 = zeros(16, 8);\nb1 = zeros(16, 1);\nh = tanh(W1 * x + b1);\nW2 = zeros(4, 16);\nb2 = zeros(4, 1);\nyhat = W2 * h + b2;\n',
                note: 'Two dense layers, weights and activations tracked end to end.',
            },
        ],
    },
    {
        group: 'Defense',
        items: [
            {
                label: 'Alpha-beta tracker',
                code: 'dt = 1.0;\nalpha = 0.85;\nbeta = 0.005;\nx = [0; 0];\nz = 1000;\nfor k = 1:10\n    xp = [x(1) + dt * x(2); x(2)];\n    rres = z - xp(1);\n    x = [xp(1) + alpha * rres; xp(2) + (beta / dt) * rres];\nend\n',
                note: 'The classic two-gain radar tracker on scalar measurements.',
            },
            {
                label: 'Ballistic trajectory',
                code: 'p = [0; 0];\nv = [120; 180];\ng = [0; -9.81];\nc = 0.002;\ndt = 0.05;\nfor k = 1:100\n    a = g - c * norm(v) * v;\n    v = v + dt * a;\n    p = p + dt * v;\nend\n',
                note: 'Drag-limited projectile flight integrated step by step.',
            },
            {
                label: 'Beamforming',
                code: "X = zeros(8, 64);\nw = ones(8, 1) / 8;\ny = w' * X;\np = (y * y') / 64;\n",
                note: 'Delay-and-sum weights applied across an array snapshot.',
            },
            {
                label: 'CV target tracker',
                code: "dt = 0.1;\nF = [1, 0, dt, 0; 0, 1, 0, dt; 0, 0, 1, 0; 0, 0, 0, 1];\nH = [1, 0, 0, 0; 0, 1, 0, 0];\nQ = 0.01 * eye(4);\nRm = 0.5 * eye(2);\nx = zeros(4, 1);\nP = eye(4);\nz = [10; 5];\nfor k = 1:20\n    x = F * x;\n    P = F * P * F' + Q;\n    S = H * P * H' + Rm;\n    K = P * H' * inv(S);\n    x = x + K * (z - H * x);\n    P = (eye(4) - K * H) * P;\nend\n",
                note: 'Predict and correct cycles of a constant-velocity tracker.',
            },
        ],
    },
    {
        group: 'Numerical methods',
        items: [
            {
                label: 'FFT spectrum',
                code: 't = linspace(0, 1, 128);\ns = sin(2 * pi * 8 * t) + 0.5 * sin(2 * pi * 20 * t);\nS = fft(s);\nm = abs(S);\n',
                note: 'A two-tone signal and its spectrum, lengths preserved end to end.',
            },
            {
                label: 'Finite differences',
                code: 't = linspace(0, 1, 101);\nf = sin(t);\ndf = (f(2:101) - f(1:100)) / 0.01;\n',
                note: 'A forward difference from two shifted slices.',
            },
            {
                label: 'Forward Euler',
                code: 'A = [0, 1; -4, 0];\ny = [1; 0];\nh = 0.01;\nfor k = 1:200\n    y = y + h * (A * y);\nend\n',
                note: 'Forward Euler marching a two-state oscillator.',
            },
            {
                label: 'Gradient descent',
                code: 'Q = [3, 1; 1, 2];\nb = [1; 1];\nx = zeros(2, 1);\nalpha = 0.1;\nfor k = 1:50\n    g = Q * x - b;\n    x = x - alpha * g;\nend\nr = Q * x - b;\n',
                note: 'Steepest descent on a quadratic, fifty steps with steady shapes.',
            },
            {
                label: 'Heat equation',
                code: 'u = zeros(50, 1);\nu(25) = 1;\nr = 0.4;\nfor k = 1:100\n    u(2:49) = u(2:49) + r * (u(1:48) - 2 * u(2:49) + u(3:50));\nend\n',
                note: 'An explicit stencil sweep over the interior, slice lengths folded from the ranges.',
            },
            {
                label: 'KKT block assembly',
                code: "Q = [2, 0, 0; 0, 2, 0; 0, 0, 2];\nA = [1, 1, 0; 0, 1, 1];\nK = [Q, A'; A, zeros(2, 2)];\nrhs = [zeros(3, 1); ones(2, 1)];\nsol = inv(K) * rhs;\n",
                note: 'Four blocks concatenated into one saddle-point system.',
            },
            {
                label: 'Markov chain',
                code: "P = [0.9, 0.1, 0; 0.2, 0.7, 0.1; 0.1, 0.2, 0.7];\np = [1; 0; 0];\nfor k = 1:100\n    p = P' * p;\nend\n",
                note: 'A transition matrix driven to its steady state.',
            },
            {
                label: "Newton's method",
                code: 'x = [1; 1];\nfor k = 1:8\n    f = [x(1)^2 + x(2)^2 - 4; x(1) * x(2) - 1];\n    J = [2 * x(1), 2 * x(2); x(2), x(1)];\n    x = x - inv(J) * f;\nend\n',
                note: 'Newton iterations on a two-variable system, Jacobian rebuilt each pass.',
            },
            {
                label: 'Polynomial interpolation',
                code: 't = [0; 1; 2; 3];\ny = [1; 2; 0; 5];\nV = [ones(4, 1), t, t.^2, t.^3];\nc = inv(V) * y;\n',
                note: 'A cubic through four points by solving the Vandermonde system.',
            },
            {
                label: 'Power iteration',
                code: "A = [2, 1; 1, 3];\nv = [1; 0];\nfor k = 1:20\n    w = A * v;\n    v = w / norm(w);\nend\nlambda = v' * A * v;\n",
                note: 'Twenty power steps converging on the dominant eigenvector.',
            },
            {
                label: 'Trapezoid rule',
                code: "h = pi / 100;\nt = linspace(0, pi, 101);\nf = sin(t);\nw = [0.5, ones(1, 99), 0.5];\nI = h * (w * f');\n",
                note: 'Composite trapezoid rule applied as a dot product.',
            },
        ],
    },
    {
        group: 'Option demos',
        items: [
            {
                label: 'Strict mode',
                code: "A = zeros(3, 3);\nr = A + 'error';\n",
                note: 'Turn on Strict mode to see this warning.',
            },
            {
                // The fixpoint toggle changes the inferred shape of x here:
                // iterated analysis widens the growing dimension, single-pass
                // does not.
                label: 'Fixpoint loops',
                code: 'x = zeros(1, 3);\nfor i = 1:5\n    x = [x, i];\nend\ny = x * ones(4, 1);\n',
                note: 'Toggle Fixpoint to watch the shape of x change. Turn on Strict mode to see the warning.',
            },
            {
                label: 'Shape annotations',
                code: "t = 0:0.1:6.2;\ns = sin(t);\nM = [s; cos(t)];\np = M * M';\n",
                note: 'Every inferred shape appears inline. Toggle Shape annotations to hide them.',
            },
            {
                label: 'Range-length shapes',
                code: 'buf = zeros(1, 8);\nfor k = 1:4\n    if k > 2\n        buf = [buf, k];\n    end\nend\ntotal = sum(buf);\n',
                note: 'With Fixpoint on, the conditionally grown buffer gets an interval shape. Strict mode adds the reassignment warning.',
            },
        ],
    },
];

const EXAMPLES: Example[] = EXAMPLE_GROUPS.flatMap(g => g.items);

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
// Bootstrap
// ---------------------------------------------------------------------------

function main(): void {
    const editorHost = document.getElementById('pg-editor');
    const panel = document.getElementById('pg-panel');
    const strictBox = document.getElementById('pg-strict') as HTMLInputElement | null;
    const fixpointBox = document.getElementById('pg-fixpoint') as HTMLInputElement | null;
    const annotationsBox = document.getElementById('pg-annotations') as HTMLInputElement | null;
    const liveBox = document.getElementById('pg-live') as HTMLInputElement | null;
    const analyzeBtn = document.getElementById('pg-analyze') as HTMLButtonElement | null;
    const exampleSelect = document.getElementById('pg-examples') as HTMLSelectElement | null;

    if (!editorHost || !panel || !strictBox || !fixpointBox || !annotationsBox
        || !liveBox || !analyzeBtn || !exampleSelect) return;

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
        all.textContent = 'All categories';
        categorySelect.appendChild(all);
        for (const g of EXAMPLE_GROUPS) {
            const opt = document.createElement('option');
            opt.value = g.group;
            opt.textContent = g.group;
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
            og.label = g.group;
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

    const showNote = (example: Example | undefined) => {
        if (!noteEl) return;
        const note = example && example.note ? example.note : '';
        noteEl.textContent = note;
        noteEl.hidden = note === '';
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

    const state = EditorState.create({
        doc: EXAMPLES[0].code,
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
                if (update.docChanged && liveBox.checked) scheduleAnalysis(update.view);
            }),
        ],
    });

    const view = new EditorView({ state, parent: editorHost });

    for (const box of [strictBox, fixpointBox, annotationsBox]) {
        box.addEventListener('change', () => runAnalysis(view, panel, options()));
    }

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
        showNote(example);
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: example.code } });
        runAnalysis(view, panel, options());
    });

    // Analyze once on load.
    runAnalysis(view, panel, options());
}

main();
