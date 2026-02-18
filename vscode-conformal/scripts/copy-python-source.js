#!/usr/bin/env node
/**
 * Copies Python source from repo root into bundled/ for VSIX packaging.
 * Run automatically via vscode:prepublish hook.
 */

const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '../..');
const bundledDir = path.resolve(__dirname, '../bundled');

const dirsToCopy = ['analysis', 'frontend', 'ir', 'lsp', 'runtime'];
const filesToCopy = ['conformal.py', 'run_all_tests.py'];

function removeDir(dir) {
    if (fs.existsSync(dir)) {
        fs.rmSync(dir, { recursive: true, force: true });
    }
}

function copyRecursive(src, dest) {
    if (!fs.existsSync(src)) {
        console.error(`Error: Source not found: ${src}`);
        process.exit(1);
    }

    const stats = fs.statSync(src);

    if (stats.isDirectory()) {
        // Skip __pycache__
        if (path.basename(src) === '__pycache__') {
            return;
        }

        if (!fs.existsSync(dest)) {
            fs.mkdirSync(dest, { recursive: true });
        }

        const entries = fs.readdirSync(src);
        for (const entry of entries) {
            copyRecursive(path.join(src, entry), path.join(dest, entry));
        }
    } else {
        // Skip .pyc files
        if (src.endsWith('.pyc')) {
            return;
        }

        fs.copyFileSync(src, dest);
    }
}

console.log('Copying Python source to bundled/...');

// Clean and recreate bundled/
removeDir(bundledDir);
fs.mkdirSync(bundledDir, { recursive: true });

// Copy directories
for (const dir of dirsToCopy) {
    const src = path.join(repoRoot, dir);
    const dest = path.join(bundledDir, dir);
    console.log(`  Copying ${dir}/`);
    copyRecursive(src, dest);
}

// Copy individual files
for (const file of filesToCopy) {
    const src = path.join(repoRoot, file);
    const dest = path.join(bundledDir, file);
    console.log(`  Copying ${file}`);
    if (!fs.existsSync(src)) {
        console.error(`Error: File not found: ${src}`);
        process.exit(1);
    }
    fs.copyFileSync(src, dest);
}

console.log('Python source copied successfully.');
