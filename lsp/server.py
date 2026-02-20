"""LSP server for Conformal MATLAB shape analyzer."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse, unquote

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from frontend.pipeline import parse_matlab
from frontend.lower_ir import lower_program
from analysis import analyze_program_ir
from analysis.context import AnalysisContext
from analysis.workspace import scan_workspace
from analysis.builtins import KNOWN_BUILTINS
from runtime.env import Env
from ir.ir import Program
from analysis.diagnostics import Diagnostic as ConformalDiagnostic, STRICT_ONLY_CODES
from lsp.diagnostics import to_lsp_diagnostic
from lsp.hover import get_hover
from lsp.code_actions import code_actions_for_diagnostic
from lsp.symbols import get_document_symbols

logger = logging.getLogger(__name__)


@dataclass
class AnalysisCache:
    """Cache for last-good analysis results per document."""

    env: Env
    diagnostics: list[ConformalDiagnostic]
    source_hash: str
    settings_hash: str  # Hash of settings used during analysis
    ir_prog: Optional[Program] = None
    function_registry: Optional[dict] = None
    external_functions: Optional[dict] = None


# Global cache: URI -> AnalysisCache
analysis_cache: Dict[str, AnalysisCache] = {}

# Debouncing: URI -> asyncio.Task
debounce_tasks: Dict[str, asyncio.Task] = {}

# Server settings (updated via workspace/didChangeConfiguration or initializationOptions)
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


def _compute_settings_hash() -> str:
    """Compute hash of current server settings for cache validation."""
    settings_str = f"{server_settings['fixpoint']}{server_settings['strict']}"
    return hashlib.sha256(settings_str.encode("utf-8")).hexdigest()


def uri_to_path(uri: str) -> Path:
    """Convert file:// URI to filesystem Path.

    Args:
        uri: File URI (e.g., file:///path/to/file.m)

    Returns:
        Path object for the file
    """
    parsed = urlparse(uri)
    # Unquote percent-encoded characters
    path_str = unquote(parsed.path)
    return Path(path_str)


def _validate(ls: LanguageServer, uri: str, source: str, force: bool = False) -> None:
    """Analyze MATLAB source and publish diagnostics.

    Args:
        ls: Language server instance
        uri: Document URI
        source: Document source text
        force: If True, bypass cache and force re-analysis
    """
    start_time = time.time()
    logger.info("Analyzing %s", uri)

    source_hash = _compute_hash(source)
    settings_hash = _compute_settings_hash()
    source_lines = source.split("\n")

    # Check cache: skip re-analysis if source and settings unchanged
    if not force and uri in analysis_cache:
        cached = analysis_cache[uri]
        if cached.source_hash == source_hash and cached.settings_hash == settings_hash:
            # Re-publish cached diagnostics
            lsp_diagnostics = [
                to_lsp_diagnostic(diag, source_lines, uri) for diag in cached.diagnostics
            ]
            ls.text_document_publish_diagnostics(
                types.PublishDiagnosticsParams(uri=uri, diagnostics=lsp_diagnostics)
            )
            logger.info("Cache hit for %s (source unchanged)", uri)
            return

    try:
        # Scan workspace for external functions
        file_path = uri_to_path(uri)
        ext = scan_workspace(file_path.parent, exclude=file_path.name)
        ctx = AnalysisContext(
            fixpoint=bool(server_settings["fixpoint"]),
            external_functions=ext
        )

        # Parse and analyze
        syntax_ast = parse_matlab(source)
        ir_prog = lower_program(syntax_ast)
        env, warnings = analyze_program_ir(ir_prog, fixpoint=ctx.fixpoint, ctx=ctx)

        # Filter low-confidence warnings in default mode
        if not server_settings["strict"]:
            warnings = [w for w in warnings if w.code not in STRICT_ONLY_CODES]

        # Convert to LSP diagnostics
        lsp_diagnostics = [
            to_lsp_diagnostic(diag, source_lines, uri) for diag in warnings
        ]

        # Publish diagnostics
        ls.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(uri=uri, diagnostics=lsp_diagnostics)
        )

        # Update cache (successful analysis)
        analysis_cache[uri] = AnalysisCache(
            env=env,
            diagnostics=warnings,
            source_hash=source_hash,
            settings_hash=settings_hash,
            ir_prog=ir_prog,
            function_registry=ctx.function_registry,
            external_functions=ctx.external_functions
        )

        elapsed = time.time() - start_time
        logger.info("Analysis complete: %s (%.3fs, %d diagnostics)", uri, elapsed, len(warnings))

    except SyntaxError as e:
        # Syntax error: user's mistake
        error_message = f"Syntax error: {str(e)}"
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
        logger.error("Analysis failed for %s: %s", uri, e, exc_info=True)

    except Exception as e:
        # Internal error: analyzer bug
        error_message = f"Internal error: {str(e)}"
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
        logger.error("Analysis failed for %s: %s", uri, e, exc_info=True)


@server.feature(types.INITIALIZE)
def initialize(ls: LanguageServer, params: types.InitializeParams):
    """Handle initialize request: apply initialization options."""
    logger.info("Server initialized")

    # Read initialization options if provided
    if params.initialization_options and isinstance(params.initialization_options, dict):
        if "fixpoint" in params.initialization_options:
            server_settings["fixpoint"] = bool(params.initialization_options["fixpoint"])
        if "strict" in params.initialization_options:
            server_settings["strict"] = bool(params.initialization_options["strict"])
        if "analyzeOnChange" in params.initialization_options:
            server_settings["analyze_on_change"] = bool(params.initialization_options["analyzeOnChange"])


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams):
    """Handle document open: analyze immediately."""
    _validate(ls, params.text_document.uri, params.text_document.text)


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
async def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams):
    """Handle document save: analyze immediately (no debounce)."""
    # Parse cache is content-addressed (auto-invalidates on file changes)

    doc = ls.workspace.get_text_document(params.text_document.uri)
    _validate(ls, params.text_document.uri, doc.source)

    # Cross-file diagnostic invalidation: re-analyze sibling files
    saved_path = uri_to_path(params.text_document.uri)
    saved_dir = saved_path.parent

    for cached_uri in list(analysis_cache.keys()):
        # Skip the saved file itself
        if cached_uri == params.text_document.uri:
            continue

        # Check if cached file is in the same directory
        try:
            cached_path = uri_to_path(cached_uri)
            if cached_path.parent == saved_dir:
                # Re-analyze with force=True (source unchanged, but external deps may have changed)
                other_doc = ls.workspace.get_text_document(cached_uri)
                _validate(ls, cached_uri, other_doc.source, force=True)
        except Exception:
            # Document may no longer be open; skip
            pass


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
    return get_hover(
        cached.env,
        doc.source,
        params.position.line,
        params.position.character,
        function_registry=cached.function_registry,
        builtins_set=KNOWN_BUILTINS,
        external_functions=cached.external_functions
    )


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


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(
    ls: LanguageServer, params: types.DocumentSymbolParams
) -> Optional[list[types.DocumentSymbol]]:
    """Handle document symbol request: return function definitions for outline view."""
    uri = params.text_document.uri

    # Check if we have IR cached
    if uri not in analysis_cache or analysis_cache[uri].ir_prog is None:
        return None

    try:
        doc = ls.workspace.get_text_document(uri)
        source_lines = doc.source.split("\n")
        cached = analysis_cache[uri]

        symbols = get_document_symbols(cached.ir_prog, source_lines)
        return symbols if symbols else None
    except Exception:
        return None


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
