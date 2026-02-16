"""LSP server for Conformal MATLAB shape analyzer."""

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
        env, warnings = analyze_program_ir(ir_prog)

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

        # Try to republish last-good cached diagnostics if available
        # (so warnings remain visible while user is mid-typing)
        if uri in analysis_cache:
            cached = analysis_cache[uri]
            # Only use cache if source hasn't changed too much
            # For now, always publish error, cache replay is optional enhancement
            pass


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
    """Handle document change: debounce analysis by 500ms."""
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
