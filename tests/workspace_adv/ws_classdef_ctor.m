% Test: cross-file classdef constructor resolution
% MyVehicle.m is a classdef file in the same directory.
% Constructing it should yield a struct with declared properties.
% EXPECT: warnings = 0
% EXPECT: v = struct{capacity: unknown, speed: unknown}

v = MyVehicle(60, 4);
