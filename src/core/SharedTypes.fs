module SharedTypes

open SymDim
open WarningCodes

// ---------------------------------------------------------------------------
// Type definitions shared across modules that need to be available before
// Context.fs in the F# compilation order.
//
// IntervalBound/Interval: extracted from Intervals.fs
// ConflictSite: extracted from Witness.fs
// ---------------------------------------------------------------------------

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
