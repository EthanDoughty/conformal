# Ethan Doughty
# matlab_parser.py
from typing import List, Tuple, Any, Union

from frontend.lexer import Token, lex, KEYWORDS
from ir.ir import (
    Program, Assign, StructAssign, CellAssign, IndexAssign, ExprStmt,
    While, For, IfChain, If, Switch, Try, Break, Continue, OpaqueStmt,
    FunctionDef, AssignMulti, Return, IndexStructAssign,
    Var, Const, StringLit, BinOp, Neg, Not, Transpose,
    FieldAccess, Lambda, FuncHandle, End, Apply, CurlyApply,
    MatrixLit, CellLit, Colon, Range, IndexExpr,
)


def extract_targets_from_tokens(tokens: List[Any]) -> List[str]:
    """Conservatively extract target variable names from raw tokens.

    Supports:
    - IDENT = ...
    - IDENT(...) = ...
    - [A, B, ...] = ... (only ID/COMMA/NEWLINE/~ in brackets)

    Args:
        tokens: List of Token objects from recovered statement

    Returns:
        List of variable names that may be assigned
    """
    if not tokens:
        return []

    targets = []

    # Simple case: IDENT = ...
    if len(tokens) >= 2 and tokens[0].kind == "ID" and tokens[1].value == "=":
        return [tokens[0].value]

    # Function-style: IDENT(...) = ...
    if len(tokens) >= 3 and tokens[0].kind == "ID" and tokens[1].value == "(":
        # Find matching ), then check for =
        depth = 0
        for i, tok in enumerate(tokens):
            if tok.value == "(":
                depth += 1
            elif tok.value == ")":
                depth -= 1
                if depth == 0 and i + 1 < len(tokens) and tokens[i + 1].value == "=":
                    return [tokens[0].value]
                break

    # Destructuring: [A, B, ...] = ...
    # Enforce strict validation: only ID, COMMA, NEWLINE, or ~ inside brackets
    if len(tokens) >= 2 and tokens[0].value == "[":
        depth = 0
        bracket_end = -1
        for i, tok in enumerate(tokens):
            if tok.value == "[":
                depth += 1
            elif tok.value == "]":
                depth -= 1
                if depth == 0:
                    bracket_end = i
                    break

        if bracket_end > 0 and bracket_end + 1 < len(tokens) and tokens[bracket_end + 1].value == "=":
            # Validate bracket contents: only ID, COMMA, NEWLINE, ~
            valid_destructuring = True
            for j in range(1, bracket_end):
                tok = tokens[j]
                if tok.kind not in {"ID", "NEWLINE"} and tok.value not in {",", "~"}:
                    valid_destructuring = False
                    break

            if valid_destructuring:
                # Extract identifiers from inside brackets
                for j in range(1, bracket_end):
                    if tokens[j].kind == "ID":
                        targets.append(tokens[j].value)
                return targets

    return []


class ParseError(Exception):
    pass

class MatlabParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0
        self.endless_functions = self._detect_endless_functions()

    def _detect_endless_functions(self) -> bool:
        """Pre-scan to detect whether this file uses end-less function style.

        Walks from the first FUNCTION token, tracking block depth and delimiter
        depth. Returns True if functions are end-less (terminated by next FUNCTION
        or EOF), False if functions end with explicit END.

        Delimiter tracking is required to ignore 'end' used as last-index inside
        (), [], {} (e.g. x(end,:)).
        """
        # Find first FUNCTION token index
        start = None
        for idx, tok in enumerate(self.tokens):
            if tok.kind == "FUNCTION":
                start = idx
                break
        if start is None:
            return False

        block_openers = {"IF", "FOR", "WHILE", "SWITCH", "TRY"}
        # Start at depth 1: we already consumed the first FUNCTION token, so we are
        # inside the outermost function body. Depth 0 would mean "between functions".
        block_depth = 1
        delim_depth = 0

        for tok in self.tokens[start + 1:]:
            if tok.kind == "EOF":
                return block_depth >= 1  # never saw an END → end-less mode

            if tok.value in ("(", "[", "{"):
                delim_depth += 1
            elif tok.value in (")", "]", "}"):
                delim_depth = max(0, delim_depth - 1)

            if delim_depth == 0:
                if tok.kind in block_openers:
                    block_depth += 1
                elif tok.kind == "END":
                    block_depth -= 1
                    if block_depth == 0:
                        return False  # END closes the outermost function → ended mode
                elif tok.kind == "FUNCTION":
                    if block_depth >= 1:
                        block_depth += 1  # nested function inside body, treat as block opener
                    else:
                        return True  # second FUNCTION at depth 0 → end-less mode

        return True  # hit end of token list without closing END → end-less mode

    # token helpers

    def current(self) -> Token:
        return self.tokens[self.i]

    def eat(self, kind: Union[str, Tuple[str, ...]]) -> Token:
        """Consume a token of the given kind or value"""
        tok = self.current()
        if isinstance(kind, tuple):
            if tok.kind not in kind and tok.value not in kind:
                raise ParseError(
                    f"Expected {kind} at {tok.pos}, found {tok.kind} {tok.value!r}"
                )
        else:
            if tok.kind != kind and tok.value != kind:
                raise ParseError(
                    f"Expected {kind} at {tok.pos}, found {tok.kind} {tok.value!r}"
                )
        self.i += 1
        return tok

    def at_end(self) -> bool:
        return self.current().kind == "EOF"

    def starts_expr(self, tok: Token) -> bool:
        return (
            tok.kind in {"NUMBER", "ID"}
            or tok.value in {"(", "-", "+", "~", "[", "{"}
            )

    def recover_to_stmt_boundary(self, start_line) -> OpaqueStmt:
        """Recover from parse error by consuming tokens until statement boundary.

        Statement boundary is `;` or NEWLINE when delimiter depth is 0.
        Tracks (), [], {} depth so newlines inside don't terminate.
        Stops before `end` at depth 0 to preserve block structure.

        Returns: OpaqueStmt with extracted targets
        """
        tokens_consumed = []
        depth = 0  # Track (), [], {} nesting

        while not self.at_end():
            tok = self.current()

            # Check for block-ending keywords at depth 0
            if depth == 0 and tok.kind in {"END", "ELSE", "ELSEIF", "CASE", "OTHERWISE", "CATCH"}:
                break

            # Check for statement boundary at depth 0 (using token kind for semicolon)
            if depth == 0:
                if tok.kind == "NEWLINE":
                    # Consume the terminator and stop
                    self.i += 1
                    break
                # Check for semicolon by value (since it's stored as token kind ";" after lexing)
                if tok.kind == ";" or tok.value == ";":
                    self.i += 1
                    break

            # Track delimiter depth using token kinds/values
            # LPAREN/RPAREN etc. are stored as their string values after lexing
            if tok.value == "(" or tok.value == "[" or tok.value == "{":
                depth += 1
            elif tok.value == ")" or tok.value == "]" or tok.value == "}":
                depth = max(0, depth - 1)

            tokens_consumed.append(tok)
            self.i += 1

        # Construct raw text from consumed tokens
        raw_text = " ".join(t.value for t in tokens_consumed)
        line, col = start_line
        targets = extract_targets_from_tokens(tokens_consumed)
        return OpaqueStmt(line=line, col=col, targets=targets, raw=raw_text)

    # top-level program

    def parse_program(self) -> Program:
        """Parse top-level program and return IR Program directly."""
        items = []
        while not self.at_end():
            while self.current().kind == "NEWLINE" or self.current().value == ";":
                if self.current().kind == "NEWLINE":
                    self.eat("NEWLINE")
                else:
                    self.eat(";")
                if self.at_end():
                    return Program(body=items)
            if self.at_end():
                break
            # Check for classdef: consume entire class block as a single opaque stmt
            if self.current().kind == "ID" and self.current().value == "classdef":
                items.append(self._consume_classdef())
                continue
            # Check for function definition
            if self.current().kind == "FUNCTION":
                items.append(self.parse_function())
            else:
                saved_i = self.i
                items.append(self.parse_stmt())
                # Guard against infinite loop: if parse_stmt didn't advance, force-skip
                if self.i == saved_i:
                    self.i += 1
        return Program(body=items)

    def _consume_classdef(self) -> OpaqueStmt:
        """Consume a classdef block as a single opaque statement.

        Tracks block depth to find the matching 'end' for the classdef.
        Does not parse internals (methods, properties, etc.).
        Returns OpaqueStmt for the entire classdef block.

        Uses paren_depth to distinguish 'end' as block closer vs. 'end' as
        array index keyword inside (), [], or {} (e.g. A(end:-1:1)).
        """
        start_tok = self.current()
        line = start_tok.line
        col = start_tok.col
        depth = 1  # classdef opens a block
        self.i += 1  # skip 'classdef'

        # Keywords that open a new block (keyword tokens)
        block_openers = {"IF", "FOR", "WHILE", "SWITCH", "TRY", "FUNCTION", "PARFOR"}
        # ID tokens that open a block inside classdef bodies
        id_block_openers = {"methods", "properties", "events", "enumeration"}

        paren_depth = 0  # tracks (), [], {} nesting

        while not self.at_end():
            tok = self.current()
            self.i += 1

            # Track delimiter nesting — end inside delimiters is an index keyword
            if tok.value in ("(", "[", "{"):
                paren_depth += 1
            elif tok.value in (")", "]", "}"):
                paren_depth = max(0, paren_depth - 1)
            elif paren_depth > 0:
                continue  # skip block depth tracking inside delimiters

            if tok.kind == "END":
                depth -= 1
                if depth == 0:
                    break
            elif tok.kind in block_openers:
                depth += 1
            elif tok.kind == "ID" and tok.value in id_block_openers:
                depth += 1

        # Skip trailing newline/semicolon
        while not self.at_end() and (self.current().kind == "NEWLINE" or self.current().kind == ";" or self.current().value == ";"):
            self.i += 1

        return OpaqueStmt(line=line, col=col, targets=[], raw="classdef")

    # function definitions

    def parse_function(self) -> FunctionDef:
        """Parse function declaration and return IR FunctionDef.

        Syntax:
          function result = name(arg1, arg2)           # single return
          function [out1, out2] = name(arg1, arg2)    # multiple returns
          function name(arg1, arg2)                   # procedure (no return)
        """
        func_tok = self.eat("FUNCTION")
        line = func_tok.line
        col = func_tok.col

        # Parse output variables (or none for procedure)
        output_vars = []

        # Check for procedure form: next token is ID followed by (
        has_parens = True
        if self.current().kind == "ID":
            # Peek ahead: is it "ID(" (procedure) or "ID =" (single return)?
            lookahead_tok = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None
            if lookahead_tok and lookahead_tok.value == "(":
                # Procedure form: function name(args)
                name = self.eat("ID").value
                self.eat("(")
            elif lookahead_tok and lookahead_tok.value == "=":
                # Single return: function result = name(args)
                output_vars.append(self.eat("ID").value)
                self.eat("=")
                name = self.eat("ID").value
                if self.current().value == "(":
                    self.eat("(")
                else:
                    has_parens = False
            elif (lookahead_tok is None
                  or lookahead_tok.kind in ("NEWLINE", "EOF", "FUNCTION")
                  or lookahead_tok.value in (";",)):
                # No-arg procedure: function name  (no parentheses)
                name = self.eat("ID").value
                has_parens = False
            else:
                raise ParseError(f"Expected '=' or '(' after function name at {self.current().pos}")
        elif self.current().value == "[":
            # Multiple outputs: function [a, b] = name(args)
            # Also handles void return: function [] = name(args)
            self.eat("[")
            if self.current().value == "]":
                # Void return: function [] = name(...)
                self.eat("]")
            else:
                output_vars.append(self.eat("ID").value)
                while self.current().value == "," or self.current().kind == "ID":
                    if self.current().value == ",":
                        self.eat(",")
                    output_vars.append(self.eat("ID").value)
                self.eat("]")
            self.eat("=")
            name = self.eat("ID").value
            if self.current().value == "(":
                self.eat("(")
            else:
                has_parens = False
        else:
            raise ParseError(f"Expected function output or name at {self.current().pos}")

        # Parse parameters
        params = []
        if has_parens:
            def _eat_param():
                if self.current().value == "~":
                    self.eat("~")
                    return "~"
                return self.eat("ID").value
            if self.current().value != ")":
                params.append(_eat_param())
                while self.current().value == ",":
                    self.eat(",")
                    params.append(_eat_param())
            self.eat(")")

        # Skip newline/semicolon after closing ) if present
        if self.current().kind == "NEWLINE" or self.current().value == ";":
            if self.current().kind == "NEWLINE":
                self.eat("NEWLINE")
            else:
                self.eat(";")

        # Parse function body
        if self.endless_functions:
            body = self.parse_block(until_kinds=("FUNCTION",))
            # No eat("END") -- body terminated by next FUNCTION or EOF
        else:
            body = self.parse_block(until_kinds=("END",))
            self.eat("END")

        return FunctionDef(line=line, col=col, name=name, params=params,
                           output_vars=output_vars, body=body)

    # statements

    def parse_stmt(self):
        tok = self.current()
        start_line = (tok.line, tok.col)
        start_pos = self.i

        try:
            if tok.kind == "FOR":
                return self.parse_for()
            elif tok.kind == "PARFOR":
                return self.parse_for(is_parfor=True)
            elif tok.kind in ("GLOBAL", "PERSISTENT"):
                return self.parse_global()
            elif tok.kind == "WHILE":
                return self.parse_while()
            elif tok.kind == "IF":
                return self.parse_if()
            elif tok.kind == "SWITCH":
                return self.parse_switch()
            elif tok.kind == "TRY":
                return self.parse_try()
            elif tok.kind == "BREAK":
                tok = self.eat("BREAK")
                return Break(line=tok.line, col=tok.col)
            elif tok.kind == "CONTINUE":
                tok = self.eat("CONTINUE")
                return Continue(line=tok.line, col=tok.col)
            elif tok.kind == "RETURN":
                tok = self.eat("RETURN")
                return Return(line=tok.line, col=tok.col)
            elif tok.kind == "FUNCTION":
                return self.parse_function()
            elif tok.kind == "NEWLINE":
                self.eat("NEWLINE")
                return ExprStmt(line=0, col=0, expr=Const(line=0, col=0, value=0.0))
            else:
                node = self.parse_simple_stmt()
                # Check for proper statement terminator
                if self.current().kind not in {"NEWLINE", "EOF"} and self.current().value != ";":
                    # Unexpected token after statement - trigger recovery
                    # Reset to start of statement and recover
                    self.i = start_pos
                    return self.recover_to_stmt_boundary(start_line)
                if self.current().kind == "NEWLINE" or self.current().value == ";":
                    if self.current().kind == "NEWLINE":
                        self.eat("NEWLINE")
                    else:
                        self.eat(";")
                return node
        except ParseError:
            # Recovery: reset to start of statement and consume tokens until boundary
            self.i = start_pos
            return self.recover_to_stmt_boundary(start_line)

    def _parse_bracket_stmt(self):
        """Parse destructuring assignment [a, b] = expr, or fall back to matrix literal."""
        saved_pos = self.i
        try:
            self.eat("[")

            def _eat_target() -> str:
                if self.current().value == "~":
                    self.eat("~")
                    return "~"
                return self.eat("ID").value

            targets = [_eat_target()]
            while self.current().value == "," or self.current().kind == "ID" or self.current().value == "~":
                if self.current().value == ",":
                    self.eat(",")
                targets.append(_eat_target())
            self.eat("]")

            if self.current().value == "=":
                eq_tok = self.eat("=")
                expr = self.parse_expr()
                return AssignMulti(line=eq_tok.line, col=eq_tok.col, targets=targets, expr=expr)
            else:
                self.i = saved_pos
                expr = self.parse_expr()
                return ExprStmt(line=expr.line, col=expr.col, expr=expr)
        except ParseError:
            self.i = saved_pos
            expr = self.parse_expr()
            return ExprStmt(line=expr.line, col=expr.col, expr=expr)

    def _parse_lhs_chain(self) -> list:
        """Greedily consume accessor chain after an ID token.

        Returns list of tuples:
          ('field', name)  -- .field or .(expr)  [dynamic yields '<dynamic>']
          ('paren', args)  -- (args)
          ('curly', args)  -- {args}
        """
        chain = []
        while True:
            if self.current().kind == "DOT":
                self.eat("DOT")
                if self.current().kind == "ID":
                    chain.append(('field', self.eat("ID").value))
                elif self.current().value == "(":
                    self.eat("(")
                    self.parse_expr()
                    self.eat(")")
                    chain.append(('field', '<dynamic>'))
                else:
                    break
            elif self.current().value == "(":
                self.eat("(")
                args = self.parse_paren_args()
                self.eat(")")
                chain.append(('paren', args))
            elif self.current().value == "{":
                self.eat("{")
                args = self.parse_paren_args()
                self.eat("}")
                chain.append(('curly', args))
            else:
                break
        return chain

    def _chain_to_expr(self, id_tok, chain):
        """Reconstruct an expression tree from an ID token and a parsed chain."""
        expr = Var(line=id_tok.line, col=id_tok.col, name=id_tok.value)
        for kind, data in chain:
            if kind == 'field':
                expr = FieldAccess(line=id_tok.line, col=id_tok.col, base=expr, field=data)
            elif kind == 'paren':
                expr = Apply(line=id_tok.line, col=id_tok.col, base=expr, args=data)
            elif kind == 'curly':
                expr = CurlyApply(line=id_tok.line, col=id_tok.col, base=expr, args=data)
        return expr

    def _classify_assignment(self, id_tok, chain, eq_tok, rhs):
        """Map a parsed LHS chain to the appropriate IR assignment node."""
        base = id_tok.value

        # Empty chain: plain assignment
        if not chain:
            return Assign(line=id_tok.line, col=id_tok.col, name=base, expr=rhs)

        # All fields: struct assignment
        if all(k == 'field' for k, _ in chain):
            fields = [d for _, d in chain]
            return StructAssign(line=eq_tok.line, col=eq_tok.col,
                                base_name=base, fields=fields, expr=rhs)

        # Single paren: indexed assignment
        if len(chain) == 1 and chain[0][0] == 'paren':
            return IndexAssign(line=eq_tok.line, col=eq_tok.col,
                               base_name=base, args=chain[0][1], expr=rhs)

        # Single curly: cell assignment
        if len(chain) == 1 and chain[0][0] == 'curly':
            return CellAssign(line=eq_tok.line, col=eq_tok.col,
                              base_name=base, args=chain[0][1], expr=rhs)

        # IndexStructAssign: first element is paren or curly, rest are all fields
        if chain[0][0] in ('paren', 'curly') and all(k == 'field' for k, _ in chain[1:]) and len(chain) > 1:
            index_kind = chain[0][0]
            index_args = chain[0][1]
            fields = [d for _, d in chain[1:]]
            return IndexStructAssign(line=eq_tok.line, col=eq_tok.col,
                                     base_name=base, index_args=index_args,
                                     index_kind=index_kind, fields=fields, expr=rhs)

        # B5 pattern: field(s), then paren/curly, then field(s) — collapse all fields
        index_pos = next((i for i, (k, _) in enumerate(chain) if k in ('paren', 'curly')), None)
        if index_pos is not None:
            prefix = chain[:index_pos]
            suffix = chain[index_pos + 1:]
            if (prefix and all(k == 'field' for k, _ in prefix)
                    and suffix and all(k == 'field' for k, _ in suffix)):
                fields = [d for _, d in prefix] + [d for _, d in suffix]
                return StructAssign(line=eq_tok.line, col=eq_tok.col,
                                    base_name=base, fields=fields, expr=rhs)

        # Fallback: unrecognised pattern, havoc base variable
        return OpaqueStmt(line=eq_tok.line, col=eq_tok.col, targets=[base], raw=[])

    def parse_simple_stmt(self):
        """Parse assignment or expression statement."""
        if self.current().value == "[":
            return self._parse_bracket_stmt()

        if self.current().kind == "ID":
            id_tok = self.eat("ID")
            chain = self._parse_lhs_chain()

            if self.current().value == "=":
                eq_tok = self.eat("=")
                rhs = self.parse_expr()
                return self._classify_assignment(id_tok, chain, eq_tok, rhs)

            # Not an assignment: reconstruct as expression and continue parsing
            expr = self._chain_to_expr(id_tok, chain)
            expr = self.parse_postfix(expr)
            expr = self.parse_expr_rest(expr, 0)
            return ExprStmt(line=expr.line, col=expr.col, expr=expr)

        expr = self.parse_expr()
        return ExprStmt(line=expr.line, col=expr.col, expr=expr)

    # control flow

    def parse_for(self, is_parfor: bool = False) -> For:
        """Parse for loop and return IR For node."""
        self.eat("PARFOR" if is_parfor else "FOR")
        var_tok = self.eat("ID")
        self.eat("=")
        # parse_expr consumes the full range expression including ':'
        # The dead-code range path is eliminated; parse_expr handles 'a:b' as BinOp
        it = self.parse_expr()
        body = self.parse_block(until_kinds=("END",))
        self.eat("END")
        return For(line=it.line, col=it.col, var=var_tok.value, it=it, body=body)

    def parse_global(self) -> OpaqueStmt:
        """Parse global/persistent declaration: collect space-separated identifiers.
        Returns OpaqueStmt directly.
        """
        kw_tok = self.eat(self.current().kind)  # eat GLOBAL or PERSISTENT
        var_names = []
        while self.current().kind == "ID":
            var_names.append(self.eat("ID").value)
        raw_text = "global " + " ".join(var_names)
        return OpaqueStmt(line=kw_tok.line, col=kw_tok.col, targets=var_names, raw=raw_text)

    def parse_while(self) -> While:
        """Parse while loop and return IR While node."""
        self.eat("WHILE")
        cond = self.parse_expr()
        body = self.parse_block(until_kinds=("END",))
        self.eat("END")
        return While(line=cond.line, col=cond.col, cond=cond, body=body)

    def parse_if(self):
        """Parse if/elseif/else and return If or IfChain IR node."""
        self.eat("IF")
        cond = self.parse_expr()
        then_body = self.parse_block(until_kinds=("ELSE", "ELSEIF", "END"))

        elseifs = []
        while self.current().kind == "ELSEIF":
            self.eat("ELSEIF")
            elif_cond = self.parse_expr()
            elif_body = self.parse_block(until_kinds=("ELSE", "ELSEIF", "END"))
            elseifs.append((elif_cond, elif_body))

        skip_stmt = ExprStmt(line=0, col=0, expr=Const(line=0, col=0, value=0.0))
        else_body = [skip_stmt]
        if self.current().kind == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_block(until_kinds=("END",))

        self.eat("END")

        if not elseifs:
            return If(line=cond.line, col=cond.col, cond=cond,
                      then_body=then_body, else_body=else_body)
        else:
            conditions = [cond] + [ec for ec, _ in elseifs]
            bodies = [then_body] + [body for _, body in elseifs]
            return IfChain(line=cond.line, col=cond.col, conditions=conditions,
                           bodies=bodies, else_body=else_body)

    def parse_switch(self) -> Switch:
        """Parse switch/case and return IR Switch node."""
        self.eat("SWITCH")
        expr = self.parse_expr()

        # Skip newline after switch expression
        if self.current().kind == "NEWLINE":
            self.eat("NEWLINE")

        cases = []
        while self.current().kind == "CASE":
            self.eat("CASE")
            case_val = self.parse_expr()
            case_body = self.parse_block(until_kinds=("CASE", "OTHERWISE", "END"))
            cases.append((case_val, case_body))

        skip_stmt = ExprStmt(line=0, col=0, expr=Const(line=0, col=0, value=0.0))
        otherwise_body = [skip_stmt]
        if self.current().kind == "OTHERWISE":
            self.eat("OTHERWISE")
            otherwise_body = self.parse_block(until_kinds=("END",))

        self.eat("END")
        return Switch(line=expr.line, col=expr.col, expr=expr,
                      cases=cases, otherwise=otherwise_body)

    def parse_try(self) -> Try:
        """Parse try/catch and return IR Try node."""
        self.eat("TRY")
        # parse_block will handle the initial newline
        try_body = self.parse_block(until_kinds=("CATCH", "END"))

        skip_stmt = ExprStmt(line=0, col=0, expr=Const(line=0, col=0, value=0.0))
        catch_body = [skip_stmt]
        if self.current().kind == "CATCH":
            self.eat("CATCH")
            # Skip optional error variable
            if self.current().kind == "ID":
                self.eat("ID")
            # parse_block will handle the newline after catch/variable
            catch_body = self.parse_block(until_kinds=("END",))

        self.eat("END")

        # Compute line from first statement in try_body
        if try_body and hasattr(try_body[0], 'line'):
            line = try_body[0].line
            col = try_body[0].col if hasattr(try_body[0], 'col') else 0
        else:
            line, col = 0, 0
        return Try(line=line, col=col, try_body=try_body, catch_body=catch_body)

    def parse_block(self, until_kinds: Tuple[str, ...]) -> List:
        """Parse block of statements until a keyword in until_kinds.
        Returns list of IR Stmt nodes (or [ExprStmt(Const(0.0))] if empty).
        """
        stmts = []
        if self.current().kind == "NEWLINE":
            self.eat("NEWLINE")
        while not self.at_end() and self.current().kind not in until_kinds:
            saved_i = self.i
            stmts.append(self.parse_stmt())
            # Guard against infinite loop: if parse_stmt didn't advance, force-skip
            if self.i == saved_i:
                self.i += 1
        if not stmts:
            return [ExprStmt(line=0, col=0, expr=Const(line=0, col=0, value=0.0))]
        return stmts

    # expressions (precedence climbing)

    PRECEDENCE = {
        "||": 0,
        "|": 1,
        "&&": 2,
        "&": 3,
        "==": 4, "~=": 4, "<": 4, "<=": 4, ">": 4, ">=": 4,
        "+": 5, "-": 5,
        "*": 6, "/": 6, ".*": 6, "./": 6, "\\": 6,
        ":": 7,
        "^": 8, ".^": 8,
    }

    def parse_expr(self, min_prec: int = 0, matrix_context: bool = False):
        """Expression grammar with precedence:
          prefix: NUMBER | STRING | ID | (expr) | -expr
          infix:  left op right

        matrix_context: when True, treat space-before-no-space-after +/- as
        end-of-expression (MATLAB matrix literal implicit separator rule).
        """
        tok = self.current()

        # prefix
        if tok.value == "+":
            # Unary plus: +expr is a no-op (identity), used in matrix literal context like [1 +2]
            self.eat("+")
            operand = self.parse_expr(self.PRECEDENCE["+"], matrix_context=matrix_context)
            left = operand  # unary + is identity
        elif tok.value == "-":
            minus_tok = self.eat("-")
            operand = self.parse_expr(self.PRECEDENCE["-"], matrix_context=matrix_context)
            left = Neg(line=minus_tok.line, col=minus_tok.col, operand=operand)
        elif tok.value == "~":
            not_tok = self.eat("~")
            operand = self.parse_expr(self.PRECEDENCE["+"], matrix_context=matrix_context)  # same precedence as unary -
            left = Not(line=not_tok.line, col=not_tok.col, operand=operand)
        elif tok.kind == "NUMBER":
            num_tok = self.eat("NUMBER")
            left = Const(line=num_tok.line, col=num_tok.col, value=float(num_tok.value))
        elif tok.kind == "STRING":
            str_tok = self.eat("STRING")
            left = StringLit(line=str_tok.line, col=str_tok.col, value=str_tok.value)
        elif tok.kind == "ID":
            id_tok = self.eat("ID")
            left = Var(line=id_tok.line, col=id_tok.col, name=id_tok.value)
            left = self.parse_postfix(left)
        elif tok.value == "(":
            self.eat("(")
            left = self.parse_expr()
            self.eat(")")
            left = self.parse_postfix(left)
        elif tok.value == "[":
            left = self.parse_matrix_literal()
            left = self.parse_postfix(left)
        elif tok.value == "{":
            left = self.parse_cell_literal()
            left = self.parse_postfix(left)
        elif tok.value == "@":
            # Anonymous function or named function handle
            at_tok = self.eat("@")
            next_tok = self.current()
            if next_tok.value == "(":
                # Anonymous function: @(params) body_expr
                self.eat("(")
                params = []
                if self.current().value != ")":
                    params.append(self.eat("ID").value)
                    while self.current().value == ",":
                        self.eat(",")
                        params.append(self.eat("ID").value)
                self.eat(")")
                body = self.parse_expr()
                left = Lambda(line=at_tok.line, col=at_tok.col, params=params, body=body)
            elif next_tok.kind == "ID":
                # Named function handle: @myFunc
                name = self.eat("ID").value
                left = FuncHandle(line=at_tok.line, col=at_tok.col, name=name)
            else:
                raise ParseError(
                    f"Expected '(' or function name after '@' at {next_tok.pos}"
                )
        elif tok.kind == "END":
            # 'end' keyword (will warn if not in indexing context during analysis)
            end_tok = self.eat("END")
            left = End(line=end_tok.line, col=end_tok.col)
        else:
            raise ParseError(
                f"Unexpected token {tok.kind} {tok.value!r} in expression at {tok.pos}"
            )

        # infix
        while True:
            tok = self.current()
            op = tok.value
            if op not in self.PRECEDENCE:
                break
            prec = self.PRECEDENCE[op]
            if prec < min_prec:
                break
            # Matrix literal spacing rule: space before +/- but no space after
            # means this is a unary sign starting a new element, not binary op.
            if matrix_context and op in ("+", "-") and self.i >= 1:
                prev_tok = self.tokens[self.i - 1]
                prev_end = prev_tok.pos + len(prev_tok.value)
                op_start = tok.pos
                space_before = op_start > prev_end
                if space_before and self.i + 1 < len(self.tokens):
                    next_tok = self.tokens[self.i + 1]
                    op_end = op_start + len(tok.value)
                    space_after = next_tok.pos > op_end
                    if not space_after:
                        break  # treat as start of next element
            op_tok = self.eat(op)
            # Right-associative: ^ and .^ use prec (not prec+1) for right operand
            right = self.parse_expr(prec if op in ("^", ".^") else prec + 1)
            left = BinOp(line=op_tok.line, col=op_tok.col, op=op, left=left, right=right)
        return left

    def parse_postfix(self, left):
        """Postfix constructs after a primary.
        - Indexing: A(i), A(i,j), A(:,j), A(i,:), A(:,:)
        - Calls: zeros(...), ones(...) (subset)
        - Apply: Unified node, dispatching deferred to analyzer
        - Field access: s.field (nested: s.a.b parsed as nested FieldAccess nodes)
        """
        while True:
            tok = self.current()

            if tok.value == "(":
                lparen_tok = self.eat("(")
                args = self.parse_paren_args()
                self.eat(")")

                # Emit unified apply node. Disambiguation happens in analyzer.
                left = Apply(line=lparen_tok.line, col=lparen_tok.col, base=left, args=args)

            elif tok.value == "{":
                # Curly indexing c{i} or c{i,j}
                lcurly_tok = self.eat("{")
                args = self.parse_paren_args()
                self.eat("}")
                left = CurlyApply(line=lcurly_tok.line, col=lcurly_tok.col, base=left, args=args)

            elif tok.kind == "TRANSPOSE":
                t_tok = self.eat("TRANSPOSE")
                left = Transpose(line=t_tok.line, col=t_tok.col, operand=left)

            elif tok.kind == "DOTOP" and tok.value == ".'":
                t_tok = self.eat("DOTOP")
                left = Transpose(line=t_tok.line, col=t_tok.col, operand=left)

            elif tok.kind == "DOT":
                # Field access: check if next token is ID (not .* or ./)
                dot_tok = self.eat("DOT")
                if self.current().kind == "ID":
                    field_name = self.eat("ID").value
                    left = FieldAccess(line=dot_tok.line, col=dot_tok.col, base=left, field=field_name)
                elif self.current().value == "(":
                    # Dynamic field access: s.(expr)
                    self.eat("(")
                    self.parse_expr()  # consume field expression but ignore value
                    self.eat(")")
                    left = FieldAccess(line=dot_tok.line, col=dot_tok.col, base=left, field="<dynamic>")
                else:
                    # Not field access (might be .* or ./ but those are DOTOP, not DOT)
                    # This shouldn't happen with current lexer, but be defensive
                    raise ParseError(f"Expected field name after '.' at {tok.pos}")

            else:
                break

        return left

    def parse_expr_rest(self, left, min_prec: int, matrix_context: bool = False):
        """Helper when the left side has already been parsed (parse_simple_stmt)"""
        while True:
            tok = self.current()
            op = tok.value
            if op not in self.PRECEDENCE:
                break
            prec = self.PRECEDENCE[op]
            if prec < min_prec:
                break
            # Matrix literal spacing rule: space before +/- but no space after
            # means this is a unary sign starting a new element, not binary op.
            if matrix_context and op in ("+", "-") and self.i >= 1:
                prev_tok = self.tokens[self.i - 1]
                prev_end = prev_tok.pos + len(prev_tok.value)
                op_start = tok.pos
                space_before = op_start > prev_end
                if space_before and self.i + 1 < len(self.tokens):
                    next_tok = self.tokens[self.i + 1]
                    op_end = op_start + len(tok.value)
                    space_after = next_tok.pos > op_end
                    if not space_after:
                        break  # treat as start of next element
            op_tok = self.eat(op)
            # Right-associative: ^ and .^ use prec (not prec+1) for right operand
            right = self.parse_expr(prec if op in ("^", ".^") else prec + 1)
            left = BinOp(line=op_tok.line, col=op_tok.col, op=op, left=left, right=right)
        return left

    def _parse_delimited_rows(self, end_token: str) -> Tuple:
        """
        Parse delimited rows for matrix or cell literals.
        Shared logic for [ ] and { } literals.
        Returns (line, col, rows) tuple.
        """
        # Get the opening token (already consumed by caller, use current position)
        cur = self.current()
        line = cur.line
        col = cur.col

        rows = []

        # Empty literal
        if self.current().value == end_token:
            return (line, col, rows)

        while True:
            # parse one row: elem (sep elem)*
            row = []

            # At least one element per row
            row.append(self.parse_expr(matrix_context=True))

            while True:
                tok = self.current()

                # explicit column separator
                if tok.value == ",":
                    self.eat(",")
                    row.append(self.parse_expr(matrix_context=True))
                    continue

                # row / end delimiters
                if tok.value in {";", end_token} or tok.kind == "NEWLINE" or tok.kind == "EOF":
                    break

                # implicit column separator (whitespace in source, skipped by lexer)
                # If the next token can start an expression, treat it as concat.
                if self.starts_expr(tok):
                    row.append(self.parse_expr(matrix_context=True))
                    continue

                break

            rows.append(row)

            # end?
            if self.current().value == end_token:
                break

            # row separator: semicolon
            if self.current().value == ";":
                self.eat(";")
                # allow trailing ; before end
                if self.current().value == end_token:
                    break
                continue

            # row separator: newline (MATLAB-style multiline literal)
            if self.current().kind == "NEWLINE":
                self.eat("NEWLINE")
                # allow trailing newline before end
                if self.current().value == end_token:
                    break
                continue

            # If we got here without ; or end or NEWLINE, it's a syntax error in literal
            tok = self.current()
            raise ParseError(
                f"Unexpected token {tok.kind} {tok.value!r} in literal at {tok.pos}"
            )

        return (line, col, rows)

    def parse_matrix_literal(self) -> MatrixLit:
        """
        Parse MATLAB-style matrix literal: [ a b, c ; d e ]
        Returns MatrixLit IR node.
        """
        lbrack = self.eat("[")
        line, col, rows = self._parse_delimited_rows("]")
        self.eat("]")
        # Use bracket token position if literal is empty
        if not rows:
            line = lbrack.line
            col = lbrack.col
        return MatrixLit(line=line, col=col, rows=rows)

    def parse_cell_literal(self) -> CellLit:
        """
        Parse MATLAB-style cell literal: { a b, c ; d e }
        Returns CellLit IR node.
        """
        lcurly = self.eat("{")
        line, col, rows = self._parse_delimited_rows("}")
        self.eat("}")
        if not rows:
            line = lcurly.line
            col = lcurly.col
        return CellLit(line=line, col=col, rows=rows)

    def parse_index_arg(self):
        """Parse a single argument inside () for indexing/calls.
        Returns IndexArg: Colon, Range, or IndexExpr.
        """
        tok = self.current()

        # : by itself
        if tok.value == ":":
            c_tok = self.eat(":")
            return Colon(line=c_tok.line, col=c_tok.col)

        # Temporarily hide : from precedence table to prevent it being consumed
        # Save original precedence
        orig_colon_prec = self.PRECEDENCE.get(":")
        if ":" in self.PRECEDENCE:
            del self.PRECEDENCE[":"]

        try:
            # Parse start expression (: won't be consumed as infix operator)
            start = self.parse_expr()

            # Check if followed by ':'
            if self.current().value == ":":
                colon_tok = self.eat(":")
                # Parse range endpoint (still with : hidden)
                range_end = self.parse_expr()
                return Range(line=colon_tok.line, col=colon_tok.col, start=start, end=range_end)

            # Bare expression — wrap in IndexExpr
            return IndexExpr(line=start.line, col=start.col, expr=start)
        finally:
            # Restore colon precedence
            if orig_colon_prec is not None:
                self.PRECEDENCE[":"] = orig_colon_prec


    def parse_paren_args(self) -> List:
        """Parse comma-separated args in (). Allows ':' as an argument.
        Skips newlines between arguments.
        Returns List[IndexArg].
        """
        args = []
        # Skip leading newlines
        while self.current().kind == "NEWLINE":
            self.eat("NEWLINE")
        if self.current().value != ")":
            args.append(self.parse_index_arg())
            while self.current().value == ",":
                self.eat(",")
                # Skip newlines after comma
                while self.current().kind == "NEWLINE":
                    self.eat("NEWLINE")
                args.append(self.parse_index_arg())
        # Skip trailing newlines
        while self.current().kind == "NEWLINE":
            self.eat("NEWLINE")
        return args


def parse_matlab(src: str) -> Program:
    """Parse MATLAB source string and return IR Program directly."""
    tokens = lex(src)
    parser = MatlabParser(tokens)
    return parser.parse_program()
