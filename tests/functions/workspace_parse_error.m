% Parser recovery handles syntax errors gracefully (OpaqueStmt),
% so the body is analyzed and result is unknown with no caller-visible warning.
% W_EXTERNAL_PARSE_ERROR only fires on true parse failures (I/O errors).
B = workspace_parse_error_helper(5);
% EXPECT: B = unknown
% EXPECT: warnings = 0
