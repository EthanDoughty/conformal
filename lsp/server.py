"""LSP server for Conformal MATLAB shape analyzer."""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from frontend.pipeline import parse_matlab
from frontend.lower_ir import lower_program
from analysis import analyze_program_ir
from runtime.env import Env
from analysis.diagnostics import Diagnostic as ConformalDiagnostic
from lsp.diagnostics import to_lsp_diagnostic
from lsp.hover import get_hover
from lsp.code_actions import code_actions_for_diagnostic


@dataclass
class AnalysisCache:
    """Cache for last-good analysis results per document."""

    env: Env
    diagnostics: list[ConformalDiagnostic]
    source_hash: str


# Global cache: URI -> AnalysisCache
analysis_cache: Dict[str, AnalysisCache] = {}

# Debouncing: URI -> asyncio.Task
debounce_tasks: Dict[str, asyncio.Task] = {}

# Server settings (updated via workspace/didChangeConfiguration)
server_settings: Dict[str, object] = {
    "fixpoint": False,
    "strict": False,
    "analyze_on_change": False,
}

# Create server instance
server = LanguageServer(
    "conformal", "v1.0", text_document_sync_kind=types.TextDocumentSyncKind.Full
)


def _compute_hash(source: str) -> str:
    """Compute hash of source text for cache validation."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _validate(ls: LanguageServer, uri: str, source: str) -> None:
    """Analyze MATLAB source and publish diagnostics.

    Args:
        ls: Language server instance
        uri: Document URI
        source: Document source text
    """
    source_hash = _compute_hash(source)
    source_lines = source.split("\n")

    try:
        # Parse and analyze
        syntax_ast = parse_matlab(source)
        ir_prog = lower_program(syntax_ast)
        env, warnings = analyze_program_ir(
            ir_prog, fixpoint=bool(server_settings["fixpoint"])
        )

        # In strict mode, check for unsupported warnings
        if server_settings["strict"]:
            unsupported = [w for w in warnings if w.code.startswith("W_UNSUPPORTED_")]
            if unsupported:
                # Add a summary error diagnostic
                warnings = list(warnings) + [
                    ConformalDiagnostic(
                        line=unsupported[0].line,
                        code="W_STRICT_MODE",
                        message=f"Strict mode: {len(unsupported)} unsupported construct(s) found",
                    )
                ]

        # Convert to LSP diagnostics
        lsp_diagnostics = [
            to_lsp_diagnostic(diag, source_lines) for diag in warnings
        ]

        # Publish diagnostics
        ls.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(uri=uri, diagnostics=lsp_diagnostics)
        )

        # Update cache (successful analysis)
        analysis_cache[uri] = AnalysisCache(
            env=env, diagnostics=warnings, source_hash=source_hash
        )

    except Exception as e:
        # Parse or analysis error: show single diagnostic at line 1
        error_message = f"Parse error: {str(e)}"
        error_diagnostic = types.Diagnostic(
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=0, character=0),
            ),
            severity=types.DiagnosticSeverity.Error,
            source="conformal",
            message=error_message,
        )

        # Publish error diagnostic
        ls.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(uri=uri, diagnostics=[error_diagnostic])
        )


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams):
    """Handle document open: analyze immediately."""
    _validate(ls, params.text_document.uri, params.text_document.text)


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
async def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams):
    """Handle document save: analyze immediately (no debounce)."""
    doc = ls.workspace.get_text_document(params.text_document.uri)
    _validate(ls, params.text_document.uri, doc.source)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
async def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams):
    """Handle document change: debounce analysis by 500ms (if enabled)."""
    if not server_settings["analyze_on_change"]:
        return

    uri = params.text_document.uri

    # Cancel existing debounce task if any
    if uri in debounce_tasks:
        debounce_tasks[uri].cancel()

    async def debounced_validate():
        """Wait 500ms then validate."""
        await asyncio.sleep(0.5)
        doc = ls.workspace.get_text_document(uri)
        _validate(ls, uri, doc.source)
        # Clean up task reference
        if uri in debounce_tasks:
            del debounce_tasks[uri]

    # Schedule new debounced validation
    task = asyncio.create_task(debounced_validate())
    debounce_tasks[uri] = task


@server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: types.HoverParams) -> Optional[types.Hover]:
    """Handle hover request: show inferred shape for variable at cursor."""
    uri = params.text_document.uri

    # Check if we have analysis results cached
    if uri not in analysis_cache:
        return None

    # Get document and cached environment
    doc = ls.workspace.get_text_document(uri)
    cached = analysis_cache[uri]

    # Get hover information
    return get_hover(cached.env, doc.source, params.position.line, params.position.character)


@server.feature(
    types.TEXT_DOCUMENT_CODE_ACTION,
    types.CodeActionOptions(code_action_kinds=[types.CodeActionKind.QuickFix]),
)
def code_action(
    ls: LanguageServer, params: types.CodeActionParams
) -> Optional[list[types.CodeAction]]:
    """Handle code action request: return quick fixes for diagnostics."""
    uri = params.text_document.uri
    try:
        doc = ls.workspace.get_text_document(uri)
    except Exception:
        return None

    source_lines = doc.source.split("\n")
    actions: list[types.CodeAction] = []

    for diagnostic in params.context.diagnostics:
        actions.extend(code_actions_for_diagnostic(diagnostic, uri, source_lines))

    return actions if actions else None


@server.feature(types.WORKSPACE_DID_CHANGE_CONFIGURATION)
def did_change_configuration(
    ls: LanguageServer, params: types.DidChangeConfigurationParams
):
    """Handle configuration changes from the client."""
    settings = getattr(params, "settings", None)
    if settings and isinstance(settings, dict):
        conformal = settings.get("conformal", {})
        if isinstance(conformal, dict):
            if "fixpoint" in conformal:
                server_settings["fixpoint"] = bool(conformal["fixpoint"])
            if "strict" in conformal:
                server_settings["strict"] = bool(conformal["strict"])
            if "analyzeOnChange" in conformal:
                server_settings["analyze_on_change"] = bool(conformal["analyzeOnChange"])

    # Re-analyze all open documents with new settings
    for uri in list(analysis_cache.keys()):
        try:
            doc = ls.workspace.get_text_document(uri)
            _validate(ls, uri, doc.source)
        except Exception:
            pass
