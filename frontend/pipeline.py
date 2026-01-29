# frontend/pipeline.py

from __future__ import annotations
from typing import Any

from frontend.matlab_parser import parse_matlab
from frontend.lower_ir import lower_program
from ir import Program as IRProgram

def parse_syntax(src: str) -> Any:
    return parse_matlab(src)

def lower_to_ir(ast: Any) -> IRProgram:
    return lower_program(ast)