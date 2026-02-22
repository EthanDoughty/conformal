import * as esbuild from 'esbuild';

const production = process.argv.includes('--minify');
const watch = process.argv.includes('--watch');

/** @type {import('esbuild').BuildOptions} */
const sharedOptions = {
    bundle: true,
    platform: 'node',
    target: 'node18',
    sourcemap: !production,
    minify: production,
    format: 'cjs',
};

// Extension client (runs in VS Code extension host)
const clientBuild = {
    ...sharedOptions,
    entryPoints: ['src/extension.ts'],
    outfile: 'out/extension.js',
    external: ['vscode'],
};

// LSP server (runs in a Node.js child process via module transport)
const serverBuild = {
    ...sharedOptions,
    entryPoints: ['src/server.ts'],
    outfile: 'out/server.js',
};

async function main() {
    if (watch) {
        const ctxClient = await esbuild.context(clientBuild);
        const ctxServer = await esbuild.context(serverBuild);
        await Promise.all([ctxClient.watch(), ctxServer.watch()]);
        console.log('[esbuild] Watching for changes...');
    } else {
        await Promise.all([
            esbuild.build(clientBuild),
            esbuild.build(serverBuild),
        ]);
        console.log('[esbuild] Build complete.');
    }
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
