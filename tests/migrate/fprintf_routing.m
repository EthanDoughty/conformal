fid = fopen('out.txt', 'w');
fprintf(fid, 'value = %d\n', v);
fprintf(2, 'warning: %s\n', w);
fprintf(1, 'progress\n');
fprintf('no handle %d\n', k);
fclose(fid);
