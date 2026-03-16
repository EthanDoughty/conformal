function y = global_persistent(x)
    global shared_state
    persistent count
    shared_state = x;
    count = count + 1;
    y = shared_state + count;
end
