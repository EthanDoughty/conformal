import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;
let statusBarItem: vscode.StatusBarItem;

function getConformalSettings(): { fixpoint: boolean; strict: boolean; analyzeOnChange: boolean } {
    const config = vscode.workspace.getConfiguration('conformal');
    return {
        fixpoint: config.get<boolean>('fixpoint', false),
        strict: config.get<boolean>('strict', false),
        analyzeOnChange: config.get<boolean>('analyzeOnChange', false),
    };
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

export function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('conformal');
    const pythonPath = config.get<string>('pythonPath', 'python3');
    const serverPath = config.get<string>('serverPath', '');

    const serverOptions: ServerOptions = {
        command: pythonPath,
        args: ['-m', 'lsp'],
        options: serverPath ? { cwd: serverPath } : {},
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'matlab' }],
        initializationOptions: getConformalSettings(),
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

    // Start client, then push initial config
    client.start().then(() => {
        client.sendNotification('workspace/didChangeConfiguration', {
            settings: { conformal: getConformalSettings() },
        });
        updateStatusBar();
    });
}

export function deactivate(): Thenable<void> | undefined {
    return client?.stop();
}
