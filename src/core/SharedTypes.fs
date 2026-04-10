// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Shared type definitions that need to be compiled before Context.fs
// in the F# build order. Contains IntervalBound/Interval (used by the
// context's value range map) and ConflictSite (used by the witness
// generator). Keeping them here breaks what would otherwise be a
// dependency cycle.

module SharedTypes

open SymDim
open WarningCodes

/// Concrete int, symbolic SymDim, or unbounded (represents -inf or +inf).
type IntervalBound =
    | Finite    of int
    | SymBound  of SymDim
    | Unbounded   // represents -infinity (lo) or +infinity (hi)


/// Closed interval [lo, hi] with optional symbolic bounds.
/// lo=Unbounded means -infinity, hi=Unbounded means +infinity.
/// Invariant: if both concrete (Finite), lo <= hi.
[<Struct>]
type Interval = {
    lo: IntervalBound
    hi: IntervalBound
}


/// Recorded dimension conflict at a warning emission point.
type ConflictSite = {
    dimA:                  Shapes.Dim
    dimB:                  Shapes.Dim
    line:                  int
    warningCode:           WarningCode
    constraintsSnapshot:   Set<string * string>
    scalarBindingsSnapshot: (string * int) list
    valueRangesSnapshot:   (string * (int * int)) list
    pathSnapshot:          (string * bool * int) list
}
