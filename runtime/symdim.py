# Ethan Doughty
# symdim.py
"""Symbolic dimension type for MATLAB static analysis.

Defines SymDim — a canonical multivariate polynomial representation
for symbolic matrix dimensions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from fractions import Fraction

# A monomial: sorted tuple of (variable_name, exponent) pairs.
# Empty tuple = constant monomial (1).
Monomial = tuple  # tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class SymDim:
    """Canonical symbolic dimension expression (multivariate polynomial).

    Sorted tuple of (monomial, coefficient) pairs.
    Canonical by construction: sorted, collected, no zero coefficients.
    """
    _terms: tuple  # tuple[tuple[Monomial, Fraction], ...]

    @staticmethod
    def const(value: int) -> "SymDim":
        """Create a constant symbolic dimension."""
        if value == 0:
            return SymDim(_terms=())
        return SymDim(_terms=(((), Fraction(value)),))

    @staticmethod
    def var(name: str) -> "SymDim":
        """Create a variable symbolic dimension."""
        return SymDim(_terms=(((((name, 1),), Fraction(1))),))

    @staticmethod
    def zero() -> "SymDim":
        """Create zero symbolic dimension."""
        return SymDim(_terms=())

    def __add__(self, other: "SymDim") -> "SymDim":
        """Add two symbolic dimensions."""
        # Merge terms from both polynomials
        term_dict = {}
        for mono, coeff in self._terms:
            term_dict[mono] = term_dict.get(mono, 0) + coeff
        for mono, coeff in other._terms:
            term_dict[mono] = term_dict.get(mono, 0) + coeff

        # Remove zero coefficients and sort
        terms = [(mono, coeff) for mono, coeff in term_dict.items() if coeff != 0]
        terms.sort(key=lambda t: self._mono_key(t[0]))
        return SymDim(_terms=tuple(terms))

    def __neg__(self) -> "SymDim":
        """Negate a symbolic dimension."""
        return SymDim(_terms=tuple((mono, -coeff) for mono, coeff in self._terms))

    def __sub__(self, other: "SymDim") -> "SymDim":
        """Subtract two symbolic dimensions."""
        return self + (-other)

    def __truediv__(self, other: "SymDim") -> "SymDim":
        """Divide two symbolic dimensions (returns rational coefficients).

        Only supports division by constants.
        """
        # Get constant divisor
        divisor = other.const_value()
        if divisor is None:
            raise ValueError("Division only supported for constant divisors")
        if divisor == 0:
            raise ValueError("Division by zero")

        # Scale all coefficients by 1/divisor
        frac_divisor = Fraction(divisor)
        new_terms = tuple((mono, coeff / frac_divisor) for mono, coeff in self._terms)
        return SymDim(_terms=new_terms)

    def __mul__(self, other: "SymDim") -> "SymDim":
        """Multiply two symbolic dimensions."""
        # Cross-product of terms
        term_dict = {}
        for mono1, coeff1 in self._terms:
            for mono2, coeff2 in other._terms:
                # Multiply monomials: combine variable exponents
                var_dict = {}
                for var, exp in mono1:
                    var_dict[var] = var_dict.get(var, 0) + exp
                for var, exp in mono2:
                    var_dict[var] = var_dict.get(var, 0) + exp
                # Sort variables to get canonical monomial
                new_mono = tuple(sorted(var_dict.items()))
                # Add to term dict
                term_dict[new_mono] = term_dict.get(new_mono, 0) + coeff1 * coeff2

        # Remove zeros and sort
        terms = [(mono, coeff) for mono, coeff in term_dict.items() if coeff != 0]
        terms.sort(key=lambda t: self._mono_key(t[0]))
        return SymDim(_terms=tuple(terms))

    def is_const(self) -> bool:
        """Check if this is a constant (no variables)."""
        return len(self._terms) == 0 or (len(self._terms) == 1 and self._terms[0][0] == ())

    def const_value(self) -> Optional[int]:
        """Return constant value if this is a constant, else None."""
        if len(self._terms) == 0:
            return 0
        if len(self._terms) == 1 and self._terms[0][0] == ():
            coeff = self._terms[0][1]
            # Only return int if denominator is 1
            if isinstance(coeff, Fraction) and coeff.denominator == 1:
                return int(coeff)
        return None

    @staticmethod
    def _mono_key(mono: Monomial) -> tuple:
        """Sort key for monomials: higher degree first, then alphabetical."""
        degree = sum(exp for var, exp in mono)
        return (-degree, mono)

    def variables(self) -> set:
        """Return set of variable names in this symbolic dimension.

        Returns:
            Set of variable name strings appearing in any term.
        """
        var_set = set()
        for mono, _ in self._terms:
            for var_name, _ in mono:
                var_set.add(var_name)
        return var_set

    def substitute(self, bindings: dict) -> "SymDim":
        """Replace bound variables with concrete values; keep free vars symbolic.

        Args:
            bindings: {var_name: int} mapping variables to concrete values.

        Returns:
            New SymDim with bound vars replaced. Free vars remain symbolic.
        """
        result = SymDim.zero()
        for mono, coeff in self._terms:
            # Start with the coefficient as a constant SymDim
            term_value = SymDim.const(1) if coeff == 1 else SymDim(_terms=(((), coeff),))
            for var_name, exp in mono:
                if var_name in bindings:
                    # Substitute concrete value: multiply term by bindings[var]^exp
                    concrete = Fraction(bindings[var_name]) ** exp
                    term_value = SymDim(_terms=tuple(
                        (m, c * concrete) for m, c in term_value._terms
                    ))
                else:
                    # Keep free variable symbolic: var_factor = SymDim.var(var_name)
                    var_factor = SymDim.var(var_name)
                    # Raise to exp by repeated multiplication
                    factor = SymDim.const(1)
                    for _ in range(exp):
                        factor = factor * var_factor
                    term_value = term_value * factor
            result = result + term_value
        return result

    def evaluate(self, bindings: dict) -> Optional[int]:
        """Evaluate to int if all variables are bound after substitution.

        Args:
            bindings: {var_name: int} mapping variables to concrete values.

        Returns:
            Integer value if fully ground, None if free variables remain or
            result is non-integer (e.g., n/2 with n=3 gives 3/2).
        """
        result = self.substitute(bindings)
        return result.const_value()

    def __str__(self) -> str:
        """Format polynomial with degree-descending, alphabetical display."""
        if len(self._terms) == 0:
            return "0"

        parts = []
        for mono, coeff in self._terms:
            # Format coefficient (handle Fraction display)
            if isinstance(coeff, Fraction):
                if coeff.denominator == 1:
                    coeff_str = str(int(coeff))
                    coeff_int = int(coeff)
                else:
                    # Rational coefficient
                    coeff_str = f"({coeff.numerator}/{coeff.denominator})"
                    coeff_int = None
            else:
                coeff_str = str(coeff)
                coeff_int = coeff

            if mono == ():
                # Constant term
                parts.append(coeff_str)
            else:
                # Variable term
                var_parts = []
                for var, exp in mono:
                    if exp == 1:
                        var_parts.append(var)
                    else:
                        var_parts.append(f"{var}^{exp}")
                var_str = "*".join(var_parts)

                # Special case for single-var single-coeff rational: n/2 instead of (1/2)*n
                if (isinstance(coeff, Fraction) and coeff.denominator != 1 and
                    len(mono) == 1 and mono[0][1] == 1 and coeff.numerator == 1):
                    # Format as "n/d" instead of "(1/d)*n"
                    parts.append(f"{var_str}/{coeff.denominator}")
                elif coeff_int == 1:
                    parts.append(var_str)
                elif coeff_int == -1:
                    parts.append(f"-{var_str}")
                else:
                    parts.append(f"{coeff_str}*{var_str}")

        # Join with + or - (handle negative coefficients)
        result = parts[0]
        for part in parts[1:]:
            if part.startswith("-"):
                result += part  # Already has minus sign
            else:
                result += "+" + part

        return result
