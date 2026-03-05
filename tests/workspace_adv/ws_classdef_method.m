% Test: cross-file classdef method dispatch
% After constructing a MyVehicle, calling obj.get_speed() should resolve
% through the method registry and return a scalar.
% EXPECT: warnings = 0
% EXPECT: v = struct{capacity: scalar, speed: scalar}
% EXPECT: s = scalar

v = MyVehicle(60, 4);
s = v.get_speed();
