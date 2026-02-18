# Ethan Doughty
# matlab_parser.py
from typing import List, Tuple, Any, Union

from frontend.lexer import Token, lex, KEYWORDS

class ParseError(Exception):
    pass

class MatlabParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0

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
            or tok.value in {"(", "-", "~", "[", "{"}
            )

    def recover_to_stmt_boundary(self, start_line: int) -> Any:
        """Recover from parse error by consuming tokens until statement boundary.

        Statement boundary is `;` or NEWLINE when delimiter depth is 0.
        Tracks (), [], {} depth so newlines inside don't terminate.
        Stops before `end` at depth 0 to preserve block structure.

        Returns: ['raw_stmt', line, tokens, raw_text]
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

        return ["raw_stmt", start_line, tokens_consumed, raw_text]

    # top-level program

    def parse_program(self) -> Any:
        """Internal form: ['seq', item1, item2, ...] where items can be statements or functions"""
        items = []
        while not self.at_end():
            while self.current().kind == "NEWLINE" or self.current().value == ";":
                if self.current().kind == "NEWLINE":
                    self.eat("NEWLINE")
                else:
                    self.eat(";")
                if self.at_end():
                    return ["seq"] + items
            if self.at_end():
                break
            # Check for function definition
            if self.current().kind == "FUNCTION":
                items.append(self.parse_function())
            else:
                items.append(self.parse_stmt())
        return ["seq"] + items

    # function definitions

    def parse_function(self) -> Any:
        """Parse function declaration.
        Internal: ['function', line, output_vars, name, params, body]

        Syntax:
          function result = name(arg1, arg2)           # single return
          function [out1, out2] = name(arg1, arg2)    # multiple returns
          function name(arg1, arg2)                   # procedure (no return)
        """
        func_tok = self.eat("FUNCTION")
        line = func_tok.line

        # Parse output variables (or none for procedure)
        output_vars = []

        # Check for procedure form: next token is ID followed by (
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
                self.eat("(")
            else:
                raise ParseError(f"Expected '=' or '(' after function name at {self.current().pos}")
        elif self.current().value == "[":
            # Multiple outputs: function [a, b] = name(args)
            self.eat("[")
            output_vars.append(self.eat("ID").value)
            while self.current().value == ",":
                self.eat(",")
                output_vars.append(self.eat("ID").value)
            self.eat("]")
            self.eat("=")
            name = self.eat("ID").value
            self.eat("(")
        else:
            raise ParseError(f"Expected function output or name at {self.current().pos}")

        # Parse parameters
        params = []
        if self.current().value != ")":
            params.append(self.eat("ID").value)
            while self.current().value == ",":
                self.eat(",")
                params.append(self.eat("ID").value)
        self.eat(")")

        # Skip newline/semicolon after closing ) if present
        if self.current().kind == "NEWLINE" or self.current().value == ";":
            if self.current().kind == "NEWLINE":
                self.eat("NEWLINE")
            else:
                self.eat(";")

        # Parse function body
        body = self.parse_block(until_kinds=("END",))
        self.eat("END")

        return ["function", line, output_vars, name, params, body]

    # statements

    def parse_stmt(self) -> Any:
        tok = self.current()
        start_line = tok.line
        start_pos = self.i

        try:
            if tok.kind == "FOR":
                return self.parse_for()
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
                return ["break", tok.line]
            elif tok.kind == "CONTINUE":
                tok = self.eat("CONTINUE")
                return ["continue", tok.line]
            elif tok.kind == "RETURN":
                tok = self.eat("RETURN")
                return ["return", tok.line]
            elif tok.kind == "NEWLINE":
                self.eat("NEWLINE")
                return ["skip"]
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

    def parse_simple_stmt(self) -> Any:
        """Parse assignment or expression statement.

        Supports:
        - ID = expr
        - [ID, ID, ...] = expr  (destructuring assignment)
        - expr (expression statement)
        """
        if self.current().value == "[":
            # Check for destructuring assignment: [a, b] = expr
            # Lookahead to distinguish from matrix literal
            saved_pos = self.i
            try:
                self.eat("[")
                # Parse ID list (IDs or ~ placeholders)
                def _eat_target() -> str:
                    if self.current().value == "~":
                        self.eat("~")
                        return "~"
                    return self.eat("ID").value
                targets = [_eat_target()]
                while self.current().value == ",":
                    self.eat(",")
                    targets.append(_eat_target())
                self.eat("]")

                # Check for =
                if self.current().value == "=":
                    # Destructuring assignment confirmed
                    eq_tok = self.eat("=")
                    expr = self.parse_expr()
                    return ["assign_multi", eq_tok.line, targets, expr]
                else:
                    # Not destructuring, backtrack and parse as matrix literal
                    self.i = saved_pos
                    expr = self.parse_expr()
                    return ["expr", expr]
            except ParseError:
                # Backtrack and parse as matrix literal
                self.i = saved_pos
                expr = self.parse_expr()
                return ["expr", expr]

        if self.current().kind == "ID":
            id_tok = self.eat("ID")
            # Check for struct assignment: ID.field.field... = expr
            if self.current().kind == "DOT":
                # Parse field chain
                fields = []
                while self.current().kind == "DOT":
                    self.eat("DOT")
                    field_name = self.eat("ID").value
                    fields.append(field_name)

                # Now check for assignment
                if self.current().value == "=":
                    eq_tok = self.eat("=")
                    expr = self.parse_expr()
                    return ["struct_assign", eq_tok.line, id_tok.value, fields, expr]
                else:
                    # Not assignment, construct field access expression and continue
                    # Build nested field_access nodes: s.a.b → field_access(field_access(var(s), a), b)
                    base = ["var", id_tok.line, id_tok.value]
                    for field in fields:
                        base = ["field_access", id_tok.line, base, field]
                    expr_tail = self.parse_expr_rest(base, 0)
                    return ["expr", expr_tail]
            # Check for cell assignment: ID{i} = expr or ID{i,j} = expr
            elif self.current().value == "{":
                # Parse curly index args
                self.eat("{")
                args = self.parse_paren_args()  # Reuse arg parser
                self.eat("}")

                # Check for assignment
                if self.current().value == "=":
                    eq_tok = self.eat("=")
                    expr = self.parse_expr()
                    return ["cell_assign", eq_tok.line, id_tok.value, args, expr]
                else:
                    # Not assignment, construct CurlyApply expression
                    base = ["var", id_tok.line, id_tok.value]
                    left = ["curly_apply", id_tok.line, base, args]
                    expr_tail = self.parse_expr_rest(left, 0)
                    return ["expr", expr_tail]
            # Check for indexed assignment: ID(i,j) = expr
            elif self.current().value == "(":
                # Parse paren index args
                self.eat("(")
                args = self.parse_paren_args()
                self.eat(")")

                # Check for assignment
                if self.current().value == "=":
                    eq_tok = self.eat("=")
                    expr = self.parse_expr()
                    return ["index_assign", eq_tok.line, id_tok.value, args, expr]
                else:
                    # Not assignment, construct Apply expression and continue
                    base = ["var", id_tok.line, id_tok.value]
                    left = ["apply", id_tok.line, base, args]
                    left = self.parse_postfix(left)  # Handle chained .field, {i}, (), '
                    expr_tail = self.parse_expr_rest(left, 0)
                    return ["expr", expr_tail]
            elif self.current().value == "=":
                self.eat("=")
                expr = self.parse_expr()
                return ["assign", id_tok.line, id_tok.value, expr]
            else:
                expr_tail = self.parse_expr_rest(["var", id_tok.line, id_tok.value], 0)
                return ["expr", expr_tail]
        else:
            expr = self.parse_expr()
            return ["expr", expr]

    # control flow

    def parse_for(self) -> Any:
        """Internal: ['for', ['var', i], ['range', start, end], body]"""
        self.eat("FOR")
        var_tok = self.eat("ID")
        self.eat("=")
        start = self.parse_expr()
        # MATLAB-style "1:10"
        if self.current().value == ":":
            self.eat(":")
            end = self.parse_expr()
            range_node = ["range", start, end]
        else:
            # fall back: treat what's after '=' as generic expr
            range_node = self.parse_expr_rest(start, 0)
        body = self.parse_block(until_kinds=("END",))
        self.eat("END")
        return ["for", ["var", var_tok.value], range_node, body]

    def parse_while(self) -> Any:
        """Internal: ['while', cond, body]"""
        self.eat("WHILE")
        cond = self.parse_expr()
        body = self.parse_block(until_kinds=("END",))
        self.eat("END")
        return ["while", cond, body]

    def parse_if(self) -> Any:
        """Internal: ['if', cond, then_body, elseifs, else_body]
        where elseifs = [[cond2, body2], [cond3, body3], ...]
        """
        self.eat("IF")
        cond = self.parse_expr()
        then_body = self.parse_block(until_kinds=("ELSE", "ELSEIF", "END"))

        elseifs = []
        while self.current().kind == "ELSEIF":
            self.eat("ELSEIF")
            elif_cond = self.parse_expr()
            elif_body = self.parse_block(until_kinds=("ELSE", "ELSEIF", "END"))
            elseifs.append([elif_cond, elif_body])

        else_body = [["skip"]]
        if self.current().kind == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_block(until_kinds=("END",))

        self.eat("END")
        return ["if", cond, then_body, elseifs, else_body]

    def parse_switch(self) -> Any:
        """Internal: ['switch', expr, cases, otherwise_body]
        where cases = [[case_val1, body1], [case_val2, body2], ...]
        """
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
            cases.append([case_val, case_body])

        otherwise_body = [["skip"]]
        if self.current().kind == "OTHERWISE":
            self.eat("OTHERWISE")
            otherwise_body = self.parse_block(until_kinds=("END",))

        self.eat("END")
        return ["switch", expr, cases, otherwise_body]

    def parse_try(self) -> Any:
        """Internal: ['try', try_body, catch_body]
        Note: Ignores optional catch variable (catch err)
        """
        self.eat("TRY")
        # parse_block will handle the initial newline
        try_body = self.parse_block(until_kinds=("CATCH", "END"))

        catch_body = [["skip"]]
        if self.current().kind == "CATCH":
            self.eat("CATCH")
            # Skip optional error variable
            if self.current().kind == "ID":
                self.eat("ID")
            # parse_block will handle the newline after catch/variable
            catch_body = self.parse_block(until_kinds=("END",))

        self.eat("END")
        return ["try", try_body, catch_body]

    def parse_block(self, until_kinds: Tuple[str, ...]) -> List[Any]:
        """Internal: [stmt1, stmt2, ...] (or [['skip']] if empty)"""
        stmts = []
        if self.current().kind == "NEWLINE":
            self.eat("NEWLINE")
        while not self.at_end() and self.current().kind not in until_kinds:
            stmts.append(self.parse_stmt())
        if not stmts:
            return [["skip"]]
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

    def parse_expr(self, min_prec: int = 0) -> Any:
        """Expression grammar with precedence:
          prefix: NUMBER | STRING | ID | (expr) | -expr
          infix:  left op right"""
        tok = self.current()

        # prefix
        if tok.value == "-":
            minus_tok = self.eat("-")
            operand = self.parse_expr(self.PRECEDENCE["-"])
            left = ["neg", minus_tok.line, operand]
        elif tok.value == "~":
            not_tok = self.eat("~")
            operand = self.parse_expr(self.PRECEDENCE["+"])  # same precedence as unary -
            left = ["not", not_tok.line, operand]
        elif tok.kind == "NUMBER":
            num_tok = self.eat("NUMBER")
            left = ["const", num_tok.line, float(num_tok.value)]
        elif tok.kind == "STRING":
            str_tok = self.eat("STRING")
            left = ["string", str_tok.line, str_tok.value]
        elif tok.kind == "ID":
            id_tok = self.eat("ID")
            left = ["var", id_tok.line, id_tok.value]
            left = self.parse_postfix(left)
        elif tok.value == "(":
            self.eat("(")
            left = self.parse_expr()
            self.eat(")")
        elif tok.value == "[":
            left = self.parse_matrix_literal()
        elif tok.value == "{":
            left = self.parse_cell_literal()
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
                left = ["lambda", at_tok.line, params, body]
            elif next_tok.kind == "ID":
                # Named function handle: @myFunc
                name = self.eat("ID").value
                left = ["func_handle", at_tok.line, name]
            else:
                raise ParseError(
                    f"Expected '(' or function name after '@' at {next_tok.pos}"
                )
        elif tok.kind == "END":
            # 'end' keyword (will warn if not in indexing context during analysis)
            end_tok = self.eat("END")
            left = ["end", end_tok.line]
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
            op_tok = self.eat(op)
            # Right-associative: ^ and .^ use prec (not prec+1) for right operand
            right = self.parse_expr(prec if op in ("^", ".^") else prec + 1)
            left = [op, op_tok.line, left, right]
        return left

    def parse_postfix(self, left: Any) -> Any:
        """Postfix constructs after a primary.
        - Indexing: A(i), A(i,j), A(:,j), A(i,:), A(:,:)
        - Calls: zeros(...), ones(...) (subset)
        - Apply: Unified node, dispatching deferred to analyzer
        - Field access: s.field (nested: s.a.b parsed as nested field_access nodes)
        """
        while True:
            tok = self.current()

            if tok.value == "(":
                lparen_tok = self.eat("(")
                args = self.parse_paren_args()
                self.eat(")")

                # Emit unified apply node. Disambiguation happens in analyzer.
                left = ["apply", lparen_tok.line, left, args]

            elif tok.value == "{":
                # Curly indexing c{i} or c{i,j}
                lcurly_tok = self.eat("{")
                args = self.parse_paren_args()
                self.eat("}")
                left = ["curly_apply", lcurly_tok.line, left, args]

            elif tok.kind == "TRANSPOSE":
                t_tok = self.eat("TRANSPOSE")
                left = ["transpose", t_tok.line, left]

            elif tok.kind == "DOT":
                # Field access: check if next token is ID (not .* or ./)
                dot_tok = self.eat("DOT")
                if self.current().kind == "ID":
                    field_name = self.eat("ID").value
                    left = ["field_access", dot_tok.line, left, field_name]
                else:
                    # Not field access (might be .* or ./ but those are DOTOP, not DOT)
                    # This shouldn't happen with current lexer, but be defensive
                    raise ParseError(f"Expected field name after '.' at {tok.pos}")

            else:
                break

        return left

    def parse_expr_rest(self, left: Any, min_prec: int) -> Any:
        """Helper when the left side has already been parsed (parse_simple_stmt)"""
        while True:
            tok = self.current()
            op = tok.value
            if op not in self.PRECEDENCE:
                break
            prec = self.PRECEDENCE[op]
            if prec < min_prec:
                break
            op_tok = self.eat(op)
            # Right-associative: ^ and .^ use prec (not prec+1) for right operand
            right = self.parse_expr(prec if op in ("^", ".^") else prec + 1)
            left = [op, op_tok.line, left, right]
        return left
    
    def _parse_delimited_rows(self, end_token: str) -> Tuple[int, List[List[Any]]]:
        """
        Parse delimited rows for matrix or cell literals.
        Shared logic for [ ] and { } literals.
        Returns (line, rows) tuple.
        """
        # Get the opening token (already consumed by caller, use current line)
        line = self.current().line

        rows: List[List[Any]] = []

        # Empty literal
        if self.current().value == end_token:
            return (line, rows)

        while True:
            # parse one row: elem (sep elem)*
            row: List[Any] = []

            # At least one element per row
            row.append(self.parse_expr())

            while True:
                tok = self.current()

                # explicit column separator
                if tok.value == ",":
                    self.eat(",")
                    row.append(self.parse_expr())
                    continue

                # row / end delimiters
                if tok.value in {";", end_token} or tok.kind == "NEWLINE" or tok.kind == "EOF":
                    break

                # implicit column separator (whitespace in source, skipped by lexer)
                # If the next token can start an expression, treat it as concat.
                if self.starts_expr(tok):
                    row.append(self.parse_expr())
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

        return (line, rows)

    def parse_matrix_literal(self) -> Any:
        """
        Parse MATLAB-style matrix literal: [ a b, c ; d e ]
        Internal form: ['matrix', line, rows]
        where rows is List[List[expr]]
        """
        lbrack = self.eat("[")
        line = lbrack.line
        line, rows = self._parse_delimited_rows("]")
        self.eat("]")
        return ["matrix", line, rows]

    def parse_cell_literal(self) -> Any:
        """
        Parse MATLAB-style cell literal: { a b, c ; d e }
        Internal form: ['cell', line, rows]
        where rows is List[List[expr]]
        """
        lcurly = self.eat("{")
        line = lcurly.line
        line, rows = self._parse_delimited_rows("}")
        self.eat("}")
        return ["cell", line, rows]
    
    def parse_index_arg(self) -> Any:
        """Parse a single argument inside () for indexing/calls
        : -> ['colon', line]
        a:b -> ['range', line, a, b]
        end -> ['end', line]
        end-1 -> ['binop', '-', ['end', line], 1]
        end:end -> ['range', line, ['end', line], ['end', line]]
        """
        tok = self.current()

        # : by itself
        if tok.value == ":":
            c_tok = self.eat(":")
            return ["colon", c_tok.line]

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
                return ["range", colon_tok.line, start, range_end]

            return start
        finally:
            # Restore colon precedence
            if orig_colon_prec is not None:
                self.PRECEDENCE[":"] = orig_colon_prec


    def parse_paren_args(self) -> List[Any]:
        """Parse comma-separated args in (). Allows ':' as an argument.
        Skips newlines between arguments."""
        args: List[Any] = []
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
    
def parse_matlab(src: str) -> Any:
    """src string -> internal AST"""
    tokens = lex(src)
    parser = MatlabParser(tokens)
    return parser.parse_program()
