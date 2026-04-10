// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Shape compatibility primitives used across the analyzer: reassignment
// conflict detection and the join logic for elementwise operations.

module AnalysisCore

open Shapes

/// Check if two shapes are provably incompatible for variable reassignment.
/// Only matrix vs matrix dimension conflicts matter.
let shapesDefinitelyIncompatible (oldShape: Shape) (newShape: Shape) : bool =
    match oldShape, newShape with
    | Bottom, _ | _, Bottom -> false
    | UnknownShape, _ | _, UnknownShape -> false
    | Matrix(r1, c1), Matrix(r2, c2) ->
        dimsDefinitelyConflict r1 r2 || dimsDefinitelyConflict c1 c2
    | _ -> false


/// Compute result shape for elementwise operations (+,-,.*,./).
let elementwiseResultShape (left: Shape) (right: Shape) : Shape =
    match left, right with
    | UnknownShape, _ | _, UnknownShape -> UnknownShape
    | Scalar, Scalar -> Scalar
    | Matrix(r1, c1), Matrix(r2, c2) ->
        if dimsDefinitelyConflict r1 r2 || dimsDefinitelyConflict c1 c2 then UnknownShape
        else Matrix(joinDim r1 r2, joinDim c1 c2)
    | _ -> UnknownShape


/// Check if an elementwise op has a provable dimension mismatch.
let elementwiseDefinitelyMismatch (left: Shape) (right: Shape) : bool =
    match left, right with
    | Matrix(r1, c1), Matrix(r2, c2) ->
        dimsDefinitelyConflict r1 r2 || dimsDefinitelyConflict c1 c2
    | _ -> false


/// Compute result shape for matrix multiplication.
let matmulResultShape (left: Shape) (right: Shape) : Shape =
    match left, right with
    | Scalar, Scalar -> Scalar
    | Scalar, Matrix _ -> right
    | Matrix _, Scalar -> left
    | Matrix(r1, c1), Matrix(r2, c2) ->
        if dimsDefinitelyConflict c1 r2 then UnknownShape
        else Matrix(r1, c2)
    | _ -> UnknownShape


/// Check if matrix multiplication has a provable inner dimension mismatch.
let matmulDefinitelyMismatch (left: Shape) (right: Shape) : bool =
    match left, right with
    | Matrix(_, c1), Matrix(r2, _) -> dimsDefinitelyConflict c1 r2
    | _ -> false
