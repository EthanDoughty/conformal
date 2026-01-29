# Ethan Doughty
# analysis/__init__.py

from legacy.analysis_legacy import analyze_program as analyze_program_legacy
from legacy.analysis_legacy import analyze_program as analyze_program
from .analysis_ir import analyze_program_ir

analyze_program = analyze_program_legacy

__all__ = ["analyze_program", "analyze_program_ir", "analyze_program_legacy"]