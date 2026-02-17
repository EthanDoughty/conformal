function result = workspace_return_helper(x)
    if x > 0
        result = x;
        return;
    end
    result = -x;
end
