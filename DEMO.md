# Conformal Demo

A quick walkthrough to see Conformal in action. Takes about 5 minutes.

## Setup

Install the extension from VS Code:

```
Ctrl+Shift+X → search "Conformal" → Install
```

Or from the command line:

```bash
code --install-extension EthanDoughty.conformal
```

No MATLAB, no .NET, no Python required. The extension runs the analyzer in-process.

## Demo 1: Basic dimension mismatch

Create a file called `demo.m` and paste this:

```matlab
A = zeros(3, 4);
B = zeros(5, 2);

C = A * B;
D = [A; B];
E = A + B;
```

You should see three underlines appear within a second:

- **Line 4**: `A * B` — inner dimensions 4 vs 5 don't match
- **Line 5**: `[A; B]` — column counts 4 vs 2 don't match
- **Line 6**: `A + B` — shapes 3x4 vs 5x2 don't match

Hover over any variable to see its inferred shape. `A` shows `matrix[3 x 4]`, `B` shows `matrix[5 x 2]`, and `C`/`D`/`E` show `unknown` (because the operations were invalid).

You should also see inlay hints like `: matrix[3 x 4]` appearing after the first assignment of each variable.

## Demo 2: Quick fixes

Replace `demo.m` with:

```matlab
A = ones(3, 4);
B = ones(3, 4);

C = A * B;
```

Conformal flags line 4: inner dimensions 4 vs 3. Click the lightbulb (or `Ctrl+.`) on the underline and you'll see **"Replace \* with .\* (elementwise)"**. Click it and the line becomes `C = A .* B;`, which is valid.

Now try this:

```matlab
x = [1 0 1 0];
y = [0 1 0 1];

if x && y
    disp('both true');
end
```

Conformal flags `&&` because `x` and `y` are vectors, not scalars. The quick fix offers **"Replace && with & (elementwise)"**. Same for `||` → `|`.

## Demo 3: Symbolic tracking

```matlab
function y = transform(A, x)
    y = A * x;
end

n = 10;
M = randn(n, n);
v = ones(n, 1);
result = transform(M, v);
```

No warnings. Hover over `result` and you'll see `matrix[10 x 1]`. Conformal tracked `n = 10` through `randn(n, n)` and `ones(n, 1)`, then propagated shapes through the function call.

Now break it:

```matlab
function y = transform(A, x)
    y = A * x;
end

n = 10;
M = randn(n, n);
v = ones(n + 1, 1);
result = transform(M, v);
```

Conformal flags `A * x` inside `transform` with the message: inner dimensions 10 vs 11. It traced the mismatch through the function boundary.

## Demo 4: Cross-file analysis

Create two files in the same directory:

**helper.m**:
```matlab
function C = combine(A, B)
    C = [A, B];
end
```

**main.m**:
```matlab
X = zeros(3, 4);
Y = zeros(5, 2);
Z = combine(X, Y);
```

Open `main.m`. Conformal finds `helper.m` in the same directory and analyzes the call. You'll see a warning on line 3: row counts 3 vs 5 don't match inside `combine`. Go-to-definition (`F12`) on `combine` jumps to `helper.m`.

## Demo 5: Status bar and modes

Look at the bottom-left of VS Code. You'll see something like:

- `$(warning) 2 warnings` — when there are problems
- `$(check) No issues` — when the file is clean

Click it to re-analyze the current file.

### Fixpoint mode

```matlab
A = zeros(2, 3);
for i = 1:10
    A = [A; zeros(1, 3)];
end
```

By default, Conformal shows `A` as `unknown` after the loop (it can't prove the final shape without running the loop). Open the command palette (`Ctrl+Shift+P`) and run **"Conformal: Toggle Fixpoint Mode"**. Now the analyzer runs the loop to convergence and infers `A`'s shape at each iteration.

## Demo 6: What it ignores (and why)

```matlab
x = 5;
y = x + 3;
z = sin(y);
A = eye(x);
b = ones(x, 1);
result = A \ b;
```

No warnings. Conformal knows `eye(x)` is 5x5, `ones(x, 1)` is 5x1, and `A \ b` is a valid 5x5 \ 5x1 = 5x1 solve. Clean code gets a clean bill of health, not noise.

## Tips

- **Inlay hints too noisy?** Set `conformal.inlayHints` to `false` in VS Code settings.
- **Want more warnings?** Enable `conformal.strict` for 11 additional low-confidence checks.
- **Have a license key?** Set `conformal.licenseKey` to unlock pro-tier checks (index bounds, division by zero, constraint conflicts, struct field validation).
- **Don't need the MathWorks extension?** Conformal bundles its own MATLAB syntax grammar.
