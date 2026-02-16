import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions } from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('conformal');
    const pythonPath = config.get<string>('pythonPath', 'python3');
    const serverPath = config.get<string>('serverPath', '');

    // Build args: python3 -m lsp (from serverPath if set, else from workspace root)
    const args = ['-m', 'lsp'];

    const serverOptions: ServerOptions = {
        command: pythonPath,
        args: args,
        options: {
            cwd: serverPath || (vscode.workspace.workspaceFolders?.[0]?.uri.fsPath),
        },
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'matlab' }],
    };

    client = new LanguageClient('conformal', 'Conformal', serverOptions, clientOptions);
    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    return client?.stop();
}
