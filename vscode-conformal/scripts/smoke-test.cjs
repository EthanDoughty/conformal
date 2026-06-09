#!/usr/bin/env node
// Smoke test: validates Fable-compiled analyzer produces correct results.
// Runs after esbuild; catches stale or missing Fable output.

const path = require('path');
const { pathToFileURL } = require('url');

const interopPath = path.resolve(__dirname, '../src/fable-out/Interop.js');

let failed = 0;
let passed = 0;

function assert(condition, label) {
    if (condition) {
        passed++;
        console.log(`  PASS: ${label}`);
    } else {
        failed++;
        console.error(`  FAIL: ${label}`);
    }
}

async function main() {
    console.log('Smoke test: loading Fable-compiled analyzer...');

    let analyzer;
    try {
        analyzer = await import(pathToFileURL(interopPath).href);
    } catch (e) {
        console.error(`FATAL: Cannot load ${interopPath}`);
        console.error(e.message);
        process.exit(1);
    }

    if (typeof analyzer.analyzeSource !== 'function') {
        console.error('FATAL: analyzeSource is not a function');
        process.exit(1);
    }

    // Test 1: Inner dimension mismatch detection
    console.log('\n1. Inner dimension mismatch');
    const r1 = analyzer.analyzeSource(
        'A = [1 2; 3 4]; B = [1 2 3; 4 5 6; 7 8 9]; C = A * B;',
        false, false, []
    );
    assert(
        r1.diagnostics.some(d => d.code === 'W_INNER_DIM_MISMATCH'),
        'W_INNER_DIM_MISMATCH detected'
    );

    // Test 2: Clean code produces no diagnostics
    console.log('\n2. Clean code');
    const r2 = analyzer.analyzeSource('x = 1;', false, false, []);
    assert(r2.diagnostics.length === 0, 'No diagnostics for clean code');

    // Test 3: W_UNKNOWN_FUNCTION fires for unrecognized calls (no longer gated)
    console.log('\n3. Unknown function detection');
    const r3 = analyzer.analyzeSource(
        'x = unknownfunc(1);', false, false, []
    );
    assert(
        r3.diagnostics.some(d => d.code === 'W_UNKNOWN_FUNCTION'),
        'W_UNKNOWN_FUNCTION detected for unrecognized call'
    );

    // Test 4: Strict mode shows additional warnings
    console.log('\n4. Strict mode');
    const r4 = analyzer.analyzeSource(
        'x = zeros(3,4); x = ones(2,2);', false, true, []
    );
    assert(
        r4.diagnostics.some(d => d.code === 'W_REASSIGN_INCOMPATIBLE'),
        'W_REASSIGN_INCOMPATIBLE present with strict=true'
    );

    // Test 5: Result structure
    console.log('\n5. Result structure');
    assert(Array.isArray(r1.diagnostics), 'diagnostics is array');
    assert(Array.isArray(r1.env), 'env is array');
    assert(Array.isArray(r1.symbols), 'symbols is array');
    assert(Array.isArray(r1.assignments), 'assignments is array');

    // Test 6: Per-diagnostic callStack crosses the Fable boundary as a JS array.
    // An F# list field here compiles to a truthy FSharpList (no .map), which made
    // server.ts throw "(d.callStack || []).map is not a function" on every diagnostic.
    console.log('\n6. Diagnostic callStack is a JS array');
    const r6 = analyzer.analyzeSource(
        'function r = scale(v)\n  r = v .* [1 2];\nend\nout = scale([1 2 3]);\n',
        true, false, []
    );
    const allDiags = [...r1.diagnostics, ...r6.diagnostics];
    assert(allDiags.length > 0, 'have diagnostics to inspect');
    assert(allDiags.every(d => Array.isArray(d.callStack)), 'every callStack is a JS array');
    assert(r6.diagnostics.some(d => d.callStack.length > 0), 'interprocedural diagnostic carries a populated call stack');
    let renderThrew = false;
    try {
        for (const d of allDiags) {
            (d.callStack || []).map(([fn, ln], i) => `${'  '.repeat(i + 1)}in ${fn}, line ${ln}`);
        }
    } catch {
        renderThrew = true;
    }
    assert(!renderThrew, 'server.ts callStack render path does not throw');

    // Summary
    console.log(`\nSmoke test: ${passed} passed, ${failed} failed`);
    process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
    console.error('FATAL:', e.message);
    process.exit(1);
});
