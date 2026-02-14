# Ethan Doughty
# matlab_parser.py
import re
from dataclasses import dataclass
from typing import List, Tuple, Any, Union

TokenKind = str

@dataclass
class Token:
    kind: TokenKind # "ID", "NUMBER", "FOR", "==", "+"
    value: str # original text
    pos: int # character offset for error messages
    line: int  # line number

# MATLAB keywords in the subset
KEYWORDS = {"for", "while", "if", "else", "end", "function"}

# Simple tokenization rules
TOKEN_SPEC = [
    ("NUMBER",   r"\d+(?:\.\d*)?"), # ints or floats
    ("ID",       r"[A-Za-z_]\w*"), # identifiers
    ("DOTOP",    r"\.\*|\./"), # element-wise ops
    ("OP",       r"==|~=|<=|>=|&&|\|\||[+\-*/<>()=,:\[\];]"),
    ("DOT",      r"\."), # standalone dot (for recovery from struct/method access)
    ("NEWLINE",  r"\n"),  # only real newlines
    ("SKIP",     r"[ \t]+"), # spaces/tabs
    ("COMMENT",  r"%[^\n]*"), # comments
    ("TRANSPOSE", r"'"), # transpose operator
    ("CURLYBRACE", r"[{}]"), # curly braces (for recovery from cell arrays)
    ("MISMATCH", r"."), # anything else is an error
]

MASTER_RE = re.compile("|".join(
    f"(?P<{name}>{pat})" for name, pat in TOKEN_SPEC
))

def lex(src: str) -> List[Token]:
    """Turn a Mini-MATLAB source string into a list of Tokens"""
    tokens: List[Token] = []
    line = 1 # Start at the first line

    for m in MASTER_RE.finditer(src):
        kind = m.lastgroup
        value = m.group()
        pos = m.start()

        if kind == "NUMBER":
            tokens.append(Token("NUMBER", value, pos, line))
        elif kind == "ID":
            if value in KEYWORDS:
                tokens.append(Token(value.upper(), value, pos, line))
            else:
                tokens.append(Token("ID", value, pos, line))
        elif kind == "DOTOP":
            tokens.append(Token(kind, value, pos, line))
        elif kind == "DOT":
            tokens.append(Token("DOT", value, pos, line))
        elif kind == "CURLYBRACE":
            tokens.append(Token(value, value, pos, line))
        elif kind == "TRANSPOSE":
            tokens.append(Token("TRANSPOSE", value, pos, line))
        elif kind == "OP":
            tokens.append(Token(value, value, pos, line))
        elif kind == "NEWLINE":
            tokens.append(Token("NEWLINE", value, pos, line))
            if value == "\n":
                line += 1
        elif kind == "SKIP" or kind == "COMMENT":
            continue
        elif kind == "MISMATCH":
            raise SyntaxError(f"Unexpected character {value!r} at {pos}")
        
    tokens.append(Token("EOF", "", len(src), line))
    return tokens

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
            or tok.value in {"(", "-", "["}
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
            if depth == 0 and tok.kind in {"END", "ELSE", "ELSEIF"}:
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
            # Recovery: consume tokens until statement boundary
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
                # Parse ID list
                targets = [self.eat("ID").value]
                while self.current().value == ",":
                    self.eat(",")
                    targets.append(self.eat("ID").value)
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
            if self.current().value == "=":
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
        """Internal: ['if', cond, then_body, else_body]"""
        self.eat("IF")
        cond = self.parse_expr()
        then_body = self.parse_block(until_kinds=("ELSE", "END"))
        else_body = [["skip"]]
        if self.current().kind == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_block(until_kinds=("END",))
        self.eat("END")
        return ["if", cond, then_body, else_body]

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
        "||": 1,
        "&&": 2,
        "==": 3, "~=": 3, "<": 3, "<=": 3, ">": 3, ">=": 3,
        "+": 4, "-": 4,
        "*": 5, "/": 5, ".*": 5, "./": 5,
        ":": 6,
    }

    def parse_expr(self, min_prec: int = 0) -> Any:
        """Expression grammar with precedence:
          prefix: NUMBER | ID | (expr) | -expr
          infix:  left op right"""
        tok = self.current()

        # prefix
        if tok.value == "-":
            minus_tok = self.eat("-")
            operand = self.parse_expr(self.PRECEDENCE["-"])
            left = ["neg", minus_tok.line, operand]
        elif tok.kind == "NUMBER":
            num_tok = self.eat("NUMBER")
            left = ["const", num_tok.line, float(num_tok.value)]
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
            right = self.parse_expr(prec + 1)
            left = [op, op_tok.line, left, right]
        return left

    def parse_postfix(self, left: Any) -> Any:
        """Postfix constructs after a primary.
        - Indexing: A(i), A(i,j), A(:,j), A(i,:), A(:,:)
        - Calls: zeros(...), ones(...) (subset)
        - Apply: Unified node, dispatching deferred to analyzer
        """
        while True:
            tok = self.current()

            if tok.value == "(":
                lparen_tok = self.eat("(")
                args = self.parse_paren_args()
                self.eat(")")

                # Emit unified apply node. Disambiguation happens in analyzer.
                left = ["apply", lparen_tok.line, left, args]

            elif tok.kind == "TRANSPOSE":
                t_tok = self.eat("TRANSPOSE")
                left = ["transpose", t_tok.line, left]

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
            right = self.parse_expr(prec + 1)
            left = [op, op_tok.line, left, right]
        return left
    
    def parse_matrix_literal(self) -> Any:
        """
        Parse MATLAB-style matrix literal: [ a b, c ; d e ]
        Internal form: ['matrix', line, rows]
        where rows is List[List[expr]]
        """
        lbrack = self.eat("[")
        line = lbrack.line

        rows: List[List[Any]] = []

        # Empty literal: []
        if self.current().value == "]":
            self.eat("]")
            return ["matrix", line, rows]

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
                if tok.value in {";", "]"} or tok.kind == "NEWLINE" or tok.kind == "EOF":
                    break

                # implicit column separator (whitespace in source, skipped by lexer)
                # If the next token can start an expression, treat it as concat.
                if self.starts_expr(tok):
                    row.append(self.parse_expr())
                    continue

                break

            rows.append(row)

            # end?
            if self.current().value == "]":
                self.eat("]")
                break

            # row separator: semicolon
            if self.current().value == ";":
                self.eat(";")
                # allow trailing ; before ]
                if self.current().value == "]":
                    self.eat("]")
                    break
                continue

            # row separator: newline (MATLAB-style multiline matrix literal)
            if self.current().kind == "NEWLINE":
                self.eat("NEWLINE")
                # allow trailing newline before ]
                if self.current().value == "]":
                    self.eat("]")
                    break
                continue

            # If we got here without ; or ] or NEWLINE, it's a syntax error in literal
            tok = self.current()
            raise ParseError(
                f"Unexpected token {tok.kind} {tok.value!r} in matrix literal at {tok.pos}"
            )

        return ["matrix", line, rows]
    
    def parse_index_arg(self) -> Any:
        """Parse a single argument inside () for indexing/calls
        : -> ['colon', line]
        a:b -> ['range', line, a, b]
        """
        tok = self.current()

        # : by itself
        if tok.value == ":":
            c_tok = self.eat(":")
            return ["colon", c_tok.line]

        # Parse a normal expression first
        start = self.parse_expr()

        # If immediately followed by ':' inside indexing args, treat it as a range
        if self.current().value == ":":
            colon_tok = self.eat(":")
            end = self.parse_expr()
            return ["range", colon_tok.line, start, end]

        return start


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
