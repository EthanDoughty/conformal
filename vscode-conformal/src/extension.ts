import * as vscode from 'vscode';
import * as path from 'path';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
} from 'vscode-languageclient/node';
import { validateLicense } from './license';

let client: LanguageClient;
let statusBarItem: vscode.StatusBarItem;
let outputChannel: vscode.OutputChannel;

// --- Configuration ---

function getConformalSettings(): { fixpoint: boolean; strict: boolean; licenseKey: string; analyzeOnChange: boolean; inlayHints: boolean } {
    const config = vscode.workspace.getConfiguration('conformal');
    return {
        fixpoint: config.get<boolean>('fixpoint', false),
        strict: config.get<boolean>('strict', false),
        licenseKey: config.get<string>('licenseKey', ''),
        analyzeOnChange: config.get<boolean>('analyzeOnChange', true),
        inlayHints: config.get<boolean>('inlayHints', true),
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
    const licenseResult = validateLicense(settings.licenseKey);
    const tierStr = licenseResult.kind === 'valid' ? ' Pro'
        : licenseResult.kind === 'grace' ? ` Pro (${licenseResult.daysLeft}d left)`
        : '';

    const modes: string[] = [];
    if (settings.fixpoint) { modes.push('fixpoint'); }
    if (settings.strict) { modes.push('strict'); }
    const modeStr = modes.length > 0 ? ` [${modes.join(', ')}]` : '';

    if (errors > 0) {
        statusBarItem.text = `$(error) Conformal${tierStr}: ${errors} error${errors > 1 ? 's' : ''}${modeStr}`;
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
    } else if (warnings > 0) {
        statusBarItem.text = `$(warning) Conformal${tierStr}: ${warnings} warning${warnings > 1 ? 's' : ''}${modeStr}`;
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
    } else {
        statusBarItem.text = `$(check) Conformal${tierStr}: No issues${modeStr}`;
        statusBarItem.backgroundColor = undefined;
    }

    statusBarItem.show();
}

// --- activate() ---

export async function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel('Conformal');
    context.subscriptions.push(outputChannel);
    outputChannel.appendLine('Conformal extension activating...');

    // Server module: bundled server.js running in a Node.js child process with IPC
    const serverModule = context.asAbsolutePath(path.join('out', 'server.js'));
    outputChannel.appendLine(`Server module: ${serverModule}`);

    const serverOptions: ServerOptions = {
        run:   { module: serverModule, transport: TransportKind.ipc },
        debug: { module: serverModule, transport: TransportKind.ipc,
                 options: { execArgv: ['--nolazy', '--inspect=6009'] } },
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
    statusBarItem.tooltip = 'Conformal MATLAB Shape Analyzer';
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
                // Force re-analysis even on already-saved files
                client.sendNotification('textDocument/didSave', {
                    textDocument: { uri: editor.document.uri.toString() },
                });
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

        vscode.commands.registerCommand('conformal.enterLicenseKey', async () => {
            const key = await vscode.window.showInputBox({
                prompt: 'Enter your Conformal Pro license key',
                placeHolder: 'CONF-...',
                ignoreFocusOut: true,
            });
            if (key) {
                const result = validateLicense(key);
                if (result.kind === 'valid' || result.kind === 'grace') {
                    const cfg = vscode.workspace.getConfiguration('conformal');
                    await cfg.update('licenseKey', key, vscode.ConfigurationTarget.Global);
                    vscode.window.showInformationMessage('Conformal Pro license activated.');
                } else {
                    const reason = result.kind === 'expired' ? 'License expired.'
                        : result.kind === 'invalid' ? `Invalid license: ${result.reason}`
                        : 'Unknown error.';
                    vscode.window.showErrorMessage(`License validation failed: ${reason}`);
                }
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
