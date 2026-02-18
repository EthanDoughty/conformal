import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as child_process from 'child_process';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;
let statusBarItem: vscode.StatusBarItem;
let outputChannel: vscode.OutputChannel;

const REQUIRED_PYTHON_MAJOR = 3;
const REQUIRED_PYTHON_MINOR = 10;
const PIP_INSTALL_TIMEOUT = 60000; // 60 seconds

// --- Platform helpers ---

function getVenvPython(venvDir: string): string {
    return process.platform === 'win32'
        ? path.join(venvDir, 'Scripts', 'python.exe')
        : path.join(venvDir, 'bin', 'python3');
}

function getVenvPip(venvDir: string): string {
    return process.platform === 'win32'
        ? path.join(venvDir, 'Scripts', 'pip.exe')
        : path.join(venvDir, 'bin', 'pip');
}

// --- Subprocess helper ---

interface ExecResult {
    stdout: string;
    stderr: string;
    exitCode: number;
}

async function execCommand(
    cmd: string,
    args: string[],
    cwd?: string,
    timeoutMs?: number
): Promise<ExecResult> {
    return new Promise((resolve) => {
        outputChannel.appendLine(`Running: ${cmd} ${args.join(' ')}`);
        const proc = child_process.execFile(
            cmd,
            args,
            { cwd, timeout: timeoutMs },
            (error, stdout, stderr) => {
                const exitCode = typeof error?.code === 'number' ? error.code : (error ? 1 : 0);
                if (stdout) outputChannel.appendLine(`stdout: ${stdout.trim()}`);
                if (stderr) outputChannel.appendLine(`stderr: ${stderr.trim()}`);
                resolve({ stdout, stderr, exitCode });
            }
        );

        if (timeoutMs) {
            setTimeout(() => {
                proc.kill();
                resolve({
                    stdout: '',
                    stderr: 'Command timed out',
                    exitCode: 124,
                });
            }, timeoutMs);
        }
    });
}

// --- Python detection ---

async function detectPython(): Promise<string> {
    const candidates = ['python3', 'python'];

    for (const cmd of candidates) {
        const result = await execCommand(cmd, ['--version']);
        if (result.exitCode !== 0) continue;

        const output = result.stdout + result.stderr;
        const match = output.match(/Python (\d+)\.(\d+)\.(\d+)/);
        if (!match) continue;

        const major = parseInt(match[1]);
        const minor = parseInt(match[2]);

        if (major === REQUIRED_PYTHON_MAJOR && minor >= REQUIRED_PYTHON_MINOR) {
            outputChannel.appendLine(`Found Python ${major}.${minor} at ${cmd}`);
            return cmd;
        } else if (major === REQUIRED_PYTHON_MAJOR) {
            throw new Error(
                `Python ${major}.${minor} found, but Conformal requires Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+. ` +
                `Please upgrade your Python installation.`
            );
        }
    }

    const installUrl =
        process.platform === 'win32'
            ? 'https://www.python.org/downloads/windows/'
            : process.platform === 'darwin'
            ? 'https://www.python.org/downloads/macos/'
            : 'https://www.python.org/downloads/';

    throw new Error(
        `Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+ not found. ` +
        `Install from ${installUrl}`
    );
}

// --- Version stamp ---

function readVersionStamp(venvDir: string): string | null {
    const stampFile = path.join(venvDir, '.conformal-version');
    if (!fs.existsSync(stampFile)) return null;
    return fs.readFileSync(stampFile, 'utf-8').trim();
}

function writeVersionStamp(venvDir: string, version: string): void {
    const stampFile = path.join(venvDir, '.conformal-version');
    fs.writeFileSync(stampFile, version, 'utf-8');
}

// --- Venv management ---

async function ensureVenv(
    context: vscode.ExtensionContext,
    progress: vscode.Progress<{ message: string }>
): Promise<string> {
    const venvDir = path.join(context.globalStoragePath, 'venv');
    const venvPython = getVenvPython(venvDir);
    const venvPip = getVenvPip(venvDir);
    const extensionVersion = context.extension.packageJSON.version;

    // Check if venv exists and version matches
    if (fs.existsSync(venvPython)) {
        const stamp = readVersionStamp(venvDir);
        if (stamp === extensionVersion) {
            outputChannel.appendLine('Using existing venv (version matches).');
            return venvPython;
        }
        outputChannel.appendLine(
            `Venv version mismatch (found ${stamp}, need ${extensionVersion}). Recreating...`
        );
    }

    // Detect Python
    progress.report({ message: 'Detecting Python...' });
    let pythonCmd: string;
    try {
        pythonCmd = await detectPython();
    } catch (err) {
        throw err;
    }

    // Remove old venv if exists
    if (fs.existsSync(venvDir)) {
        outputChannel.appendLine('Removing old venv...');
        fs.rmSync(venvDir, { recursive: true, force: true });
    }

    // Ensure globalStoragePath exists
    if (!fs.existsSync(context.globalStoragePath)) {
        fs.mkdirSync(context.globalStoragePath, { recursive: true });
    }

    // Create venv
    progress.report({ message: 'Creating Python virtual environment...' });
    outputChannel.appendLine(`Creating venv at ${venvDir}...`);
    const venvResult = await execCommand(pythonCmd, ['-m', 'venv', venvDir]);
    if (venvResult.exitCode !== 0) {
        const stderr = venvResult.stderr.toLowerCase();
        if (stderr.includes('ensurepip') || stderr.includes('not available')) {
            throw new Error(
                'Failed to create virtual environment. ' +
                'On Debian/Ubuntu, install python3-venv: sudo apt install python3-venv'
            );
        }
        throw new Error(`Failed to create virtual environment: ${venvResult.stderr}`);
    }

    // Install pygls
    progress.report({ message: 'Installing dependencies (pygls)...' });
    outputChannel.appendLine('Installing pygls...');
    const pipResult = await execCommand(
        venvPip,
        ['install', 'pygls>=2.0,<3.0'],
        undefined,
        PIP_INSTALL_TIMEOUT
    );
    if (pipResult.exitCode !== 0) {
        throw new Error(
            'Failed to install dependencies. Check your network connection and try again.'
        );
    }

    // Write version stamp
    writeVersionStamp(venvDir, extensionVersion);
    outputChannel.appendLine('Venv setup complete.');

    return venvPython;
}

// --- Configuration ---

function getConformalSettings(): { fixpoint: boolean; strict: boolean; analyzeOnChange: boolean } {
    const config = vscode.workspace.getConfiguration('conformal');
    return {
        fixpoint: config.get<boolean>('fixpoint', false),
        strict: config.get<boolean>('strict', false),
        analyzeOnChange: config.get<boolean>('analyzeOnChange', true),
    };
}

function reclassifyIfMatlab(doc: vscode.TextDocument): void {
    if (doc.fileName.endsWith('.m') && doc.languageId !== 'matlab') {
        vscode.languages.setTextDocumentLanguage(doc, 'matlab');
    }
}

function updateStatusBar(): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'matlab') {
        statusBarItem.hide();
        return;
    }

    const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
    const errors = diagnostics.filter(d => d.severity === vscode.DiagnosticSeverity.Error).length;
    const warnings = diagnostics.filter(d => d.severity === vscode.DiagnosticSeverity.Warning).length;

    const settings = getConformalSettings();
    const modes: string[] = [];
    if (settings.fixpoint) { modes.push('fixpoint'); }
    if (settings.strict) { modes.push('strict'); }
    const modeStr = modes.length > 0 ? ` [${modes.join(', ')}]` : '';

    if (errors > 0) {
        statusBarItem.text = `$(error) Conformal: ${errors} error${errors > 1 ? 's' : ''}${modeStr}`;
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
    } else if (warnings > 0) {
        statusBarItem.text = `$(warning) Conformal: ${warnings} warning${warnings > 1 ? 's' : ''}${modeStr}`;
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
    } else {
        statusBarItem.text = `$(check) Conformal: Ready${modeStr}`;
        statusBarItem.backgroundColor = undefined;
    }

    statusBarItem.show();
}

// --- activate() ---

export async function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel('Conformal');
    context.subscriptions.push(outputChannel);
    outputChannel.appendLine('Conformal extension activating...');

    const config = vscode.workspace.getConfiguration('conformal');
    const pythonPathSetting = config.get<string>('pythonPath', 'python3');
    const serverPathSetting = config.get<string>('serverPath', '');

    let pythonPath: string;
    let serverCwd: string;

    // Check for dev overrides
    const hasCustomPython = pythonPathSetting !== 'python3';
    const hasCustomServer = serverPathSetting !== '';

    if (hasCustomPython || hasCustomServer) {
        outputChannel.appendLine('Dev override detected — skipping auto-setup.');
        pythonPath = pythonPathSetting;
        serverCwd = serverPathSetting;
    } else {
        // Auto-setup: create venv and use bundled source
        try {
            pythonPath = await vscode.window.withProgress(
                {
                    location: vscode.ProgressLocation.Notification,
                    title: 'Conformal: Setting up analyzer...',
                    cancellable: false,
                },
                async (progress) => {
                    return await ensureVenv(context, progress);
                }
            );
            serverCwd = path.join(context.extensionPath, 'bundled');
            outputChannel.appendLine(`Using bundled analyzer at ${serverCwd}`);
        } catch (err) {
            const errMsg = err instanceof Error ? err.message : String(err);
            outputChannel.appendLine(`Setup failed: ${errMsg}`);

            // Show error with optional retry
            const action = await vscode.window.showErrorMessage(
                `Conformal setup failed: ${errMsg}`,
                'Retry',
                'Cancel'
            );

            if (action === 'Retry') {
                return activate(context);
            }
            return;
        }
    }

    outputChannel.appendLine(`Python: ${pythonPath}`);
    outputChannel.appendLine(`Server path: ${serverCwd || '(default)'}`);

    const serverOptions: ServerOptions = {
        command: pythonPath,
        args: ['-m', 'lsp'],
        options: serverCwd ? { cwd: serverCwd } : {},
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'matlab' }],
        initializationOptions: getConformalSettings(),
        outputChannel: outputChannel,
        connectionOptions: { maxRestartCount: 3 },
    };

    client = new LanguageClient('conformal', 'Conformal', serverOptions, clientOptions);

    // Status bar
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 0);
    statusBarItem.command = 'conformal.analyzeFile';
    statusBarItem.tooltip = 'Conformal MATLAB Shape Analyzer — click to analyze';
    context.subscriptions.push(statusBarItem);

    // Update status bar on diagnostic changes and editor switches
    context.subscriptions.push(
        vscode.languages.onDidChangeDiagnostics(() => updateStatusBar()),
        vscode.window.onDidChangeActiveTextEditor(() => updateStatusBar()),
    );

    // Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('conformal.analyzeFile', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'matlab') {
                await editor.document.save();
            }
        }),

        vscode.commands.registerCommand('conformal.toggleFixpoint', async () => {
            const cfg = vscode.workspace.getConfiguration('conformal');
            const current = cfg.get<boolean>('fixpoint', false);
            await cfg.update('fixpoint', !current, vscode.ConfigurationTarget.Workspace);
            vscode.window.showInformationMessage(`Conformal: Fixpoint mode ${!current ? 'enabled' : 'disabled'}`);
        }),

        vscode.commands.registerCommand('conformal.toggleStrict', async () => {
            const cfg = vscode.workspace.getConfiguration('conformal');
            const current = cfg.get<boolean>('strict', false);
            await cfg.update('strict', !current, vscode.ConfigurationTarget.Workspace);
            vscode.window.showInformationMessage(`Conformal: Strict mode ${!current ? 'enabled' : 'disabled'}`);
        }),

        vscode.commands.registerCommand('conformal.restartServer', async () => {
            if (client) {
                outputChannel.appendLine('Restarting server...');
                await client.stop();
                await client.start();
                vscode.window.showInformationMessage('Conformal: Server restarted');
            }
        }),
    );

    // Forward config changes to server
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(event => {
            if (event.affectsConfiguration('conformal')) {
                client.sendNotification('workspace/didChangeConfiguration', {
                    settings: { conformal: getConformalSettings() },
                });
                updateStatusBar();
            }
        }),
    );

    // Reclassify .m files that VS Code opened as Objective-C
    vscode.workspace.textDocuments.forEach(reclassifyIfMatlab);
    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument(reclassifyIfMatlab),
    );

    // Start client
    outputChannel.appendLine('Starting language server...');
    client.start().then(() => {
        outputChannel.appendLine('Language server started successfully.');
        client.sendNotification('workspace/didChangeConfiguration', {
            settings: { conformal: getConformalSettings() },
        });
        updateStatusBar();
    }).catch((err: Error) => {
        const msg = `Failed to start Conformal server: ${err.message}`;
        outputChannel.appendLine(msg);
        vscode.window.showErrorMessage(msg);
    });
}

export function deactivate(): Thenable<void> | undefined {
    return client?.stop();
}
