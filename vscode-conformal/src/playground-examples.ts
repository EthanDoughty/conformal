// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Playground example templates and the generator that turns one into
// editor text. A template's editable values (params) name assignment
// lines by their literal left-hand side; the default value is read back
// out of the template code itself, so the code and the parameter panel
// can never disagree. When variable comments are on, the generator adds
// a header line and aligned end-of-line comments from the docs map.
// Kept free of DOM and CodeMirror imports so a plain node script can
// load it and verify every generated variant against the analyzer.

export interface ExampleParam {
    key: string;    // literal left-hand side of the assignment line to edit
    label: string;  // short name shown beside the input
}

export interface Example {
    label: string;
    code: string;
    note?: string;   // shown as a muted caption while the example is selected
    params?: ExampleParam[];             // editable values, in panel order
    docs?: { [lhs: string]: string };    // comment text per assignment line
}

// Every example is verified against the analyzer before shipping: the error
// ones emit exactly the named warning, the clean ones emit nothing.
export const EXAMPLE_GROUPS: { group: string; items: Example[] }[] = [
    {
        group: 'Dimension errors',
        items: [
            {
                label: 'Matrix multiply',
                code: 'A = zeros(3,4);\nB = ones(5,6);\nC = A * B;\n',
                params: [
                    { key: 'A', label: 'Matrix A' },
                    { key: 'B', label: 'Matrix B' },
                ],
                docs: {
                    'A': 'left factor of the product',
                    'B': 'right factor of the product',
                    'C': 'needs cols of A to equal rows of B',
                },
            },
            {
                label: 'Elementwise op',
                code: 'A = zeros(2, 3);\nB = ones(3, 2);\nC = A .* B;\n',
                params: [
                    { key: 'A', label: 'Matrix A' },
                    { key: 'B', label: 'Matrix B' },
                ],
                docs: {
                    'A': 'first operand',
                    'B': 'second operand',
                    'C': 'elementwise product, shapes must match',
                },
            },
            {
                label: 'Concatenation',
                code: 'top = zeros(2, 3);\nbottom = ones(2, 4);\nM = [top; bottom];\n',
                params: [
                    { key: 'top', label: 'Top block' },
                    { key: 'bottom', label: 'Bottom block' },
                ],
                docs: {
                    'top': 'upper block of the stack',
                    'bottom': 'lower block, column count must agree',
                    'M': 'the two blocks stacked vertically',
                },
            },
            {
                label: 'Indexing',
                code: 'A = zeros(2, 2);\nb = A(3, 1);\n',
                params: [
                    { key: 'A', label: 'Matrix' },
                    { key: 'b', label: 'Element read' },
                ],
                docs: {
                    'A': 'the matrix being indexed',
                    'b': 'a read that must stay inside the bounds',
                },
            },
        ],
    },
    {
        group: 'Automotive',
        items: [
            {
                label: 'ADAS sensor fusion',
                code: 'x_cam = [12.1; 3.4];\nP_cam = [0.6, 0; 0, 0.9];\nx_rad = [12.4; 3.1];\nP_rad = [0.2, 0; 0, 1.5];\nW1 = inv(P_cam);\nW2 = inv(P_rad);\nP_f = inv(W1 + W2);\nx_f = P_f * (W1 * x_cam + W2 * x_rad);\n',
                note: 'Camera and radar estimates merged by their covariances.',
                params: [
                    { key: 'x_cam', label: 'Camera estimate' },
                    { key: 'P_cam', label: 'Camera covariance' },
                    { key: 'x_rad', label: 'Radar estimate' },
                    { key: 'P_rad', label: 'Radar covariance' },
                ],
                docs: {
                    'x_cam': 'track position from the camera',
                    'P_cam': 'camera measurement covariance',
                    'x_rad': 'track position from the radar',
                    'P_rad': 'radar measurement covariance',
                    'W1': 'information weight of the camera',
                    'W2': 'information weight of the radar',
                    'P_f': 'covariance of the fused estimate',
                    'x_f': 'covariance weighted fused position',
                },
            },
            {
                label: 'Battery RC model',
                code: 'dt = 1;\nCq = 8000;\nRrc = 0.015;\nCrc = 2400;\nA = [1, 0; 0, 1 - dt / (Rrc * Crc)];\nB = [-dt / Cq; dt / Crc];\nx = [0.9; 0];\nfor k = 1:60\n    x = A * x + B * 12;\nend\nv = 3.6 + 0.7 * x(1) - x(2);\n',
                note: 'An equivalent-circuit battery state stepped over a discharge.',
                params: [
                    { key: 'dt', label: 'Time step' },
                    { key: 'Cq', label: 'Cell capacity' },
                    { key: 'Rrc', label: 'RC resistance' },
                    { key: 'Crc', label: 'RC capacitance' },
                ],
                docs: {
                    'dt': 'step length in seconds',
                    'Cq': 'charge capacity in coulombs',
                    'Rrc': 'resistance of the RC pair in ohms',
                    'Crc': 'capacitance of the RC pair in farads',
                    'A': 'state transition for charge and RC voltage',
                    'B': 'input map for the discharge current',
                    'x': 'state of charge and RC branch voltage',
                    'v': 'terminal voltage at the end',
                },
            },
            {
                label: 'Braking distance sweep',
                code: 'v0 = linspace(10, 40, 30);\nmu_f = 0.8;\ntreact = 1.2;\nd = v0.^2 / (2 * mu_f * 9.81) + treact * v0;\nlongest = max(d);\n',
                note: 'Stopping distances over a speed sweep with reaction time.',
                params: [
                    { key: 'v0', label: 'Speed sweep' },
                    { key: 'mu_f', label: 'Friction' },
                    { key: 'treact', label: 'Reaction time' },
                ],
                docs: {
                    'v0': 'initial speeds to sweep in m/s',
                    'mu_f': 'tire to road friction coefficient',
                    'treact': 'driver reaction time in seconds',
                    'd': 'stopping distance at each speed',
                    'longest': 'worst case in the sweep',
                },
            },
            {
                label: 'Quarter-car suspension',
                code: 'ms = 300;\nmu = 40;\nks = 18000;\nku = 180000;\ncs = 1200;\nA = [0, 1, 0, 0; -ks / ms, -cs / ms, ks / ms, cs / ms; 0, 0, 0, 1; ks / mu, cs / mu, -(ks + ku) / mu, -cs / mu];\nx = [0.02; 0; 0; 0];\ndt = 0.001;\nfor k = 1:2000\n    x = x + dt * A * x;\nend\n',
                note: 'Body and wheel states through a bump in the quarter-car model.',
                params: [
                    { key: 'ms', label: 'Sprung mass' },
                    { key: 'mu', label: 'Unsprung mass' },
                    { key: 'ks', label: 'Spring rate' },
                    { key: 'cs', label: 'Damping' },
                ],
                docs: {
                    'ms': 'sprung body mass in kg',
                    'mu': 'unsprung wheel mass in kg',
                    'ks': 'suspension spring rate in N/m',
                    'ku': 'tire stiffness in N/m',
                    'cs': 'suspension damping in N s/m',
                    'A': 'state matrix for both masses',
                    'x': 'initial bump on the body, at rest',
                    'dt': 'integration step in seconds',
                },
            },
            {
                label: 'Wheel slip ratios',
                code: 'wheel = [21.8, 22.1, 20.4, 21.9];\nrw = 0.32;\nvx = 7.1;\nslip = (vx - rw * wheel) / vx;\nworst = max(slip);\n',
                note: 'Elementwise slip computation across all four wheels.',
                params: [
                    { key: 'wheel', label: 'Wheel speeds' },
                    { key: 'rw', label: 'Wheel radius' },
                    { key: 'vx', label: 'Vehicle speed' },
                ],
                docs: {
                    'wheel': 'wheel angular speeds in rad/s',
                    'rw': 'rolling radius in meters',
                    'vx': 'vehicle ground speed in m/s',
                    'slip': 'slip ratio of all four wheels at once',
                    'worst': 'largest slip of the four',
                },
            },
        ],
    },
    {
        group: 'Aviation & navigation',
        items: [
            {
                label: '3D rigid transform',
                code: 'theta = pi / 4;\nR = [cos(theta), -sin(theta), 0; sin(theta), cos(theta), 0; 0, 0, 1];\nt = [1; 2; 0];\nP = zeros(3, 25);\nmoved = R * P + t * ones(1, 25);\n',
                note: 'Rotate and translate a point cloud in one expression.',
                params: [
                    { key: 'theta', label: 'Rotation angle' },
                    { key: 't', label: 'Translation' },
                ],
                docs: {
                    'theta': 'rotation angle in radians',
                    'R': 'rotation about the vertical axis',
                    't': 'translation applied to every point',
                    'P': 'point cloud, one column per point',
                    'moved': 'the cloud rotated then translated',
                },
            },
            {
                label: 'ECEF to NED transform',
                code: 'lat = 0.68;\nlon = -1.66;\nR = [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat); -sin(lon), cos(lon), 0; -cos(lat) * cos(lon), -cos(lat) * sin(lon), -sin(lat)];\ndp = [1200; -340; 560];\nned = R * dp;\n',
                note: 'The local navigation frame built from latitude and longitude.',
                params: [
                    { key: 'lat', label: 'Latitude (rad)' },
                    { key: 'lon', label: 'Longitude (rad)' },
                    { key: 'dp', label: 'ECEF offset' },
                ],
                docs: {
                    'lat': 'geodetic latitude in radians',
                    'lon': 'longitude in radians',
                    'R': 'rotation from ECEF to north east down',
                    'dp': 'position offset in ECEF meters',
                    'ned': 'the same offset in local axes',
                },
            },
            {
                label: 'Euler DCM chain',
                code: "phi = 0.1;\ntheta = 0.05;\npsi = 1.2;\nRx = [1, 0, 0; 0, cos(phi), sin(phi); 0, -sin(phi), cos(phi)];\nRy = [cos(theta), 0, -sin(theta); 0, 1, 0; sin(theta), 0, cos(theta)];\nRz = [cos(psi), sin(psi), 0; -sin(psi), cos(psi), 0; 0, 0, 1];\nC = Rx * Ry * Rz;\nv_ned = C' * [25; 2; -1];\n",
                note: 'A 3-2-1 rotation sequence assembled and applied to a body vector.',
                params: [
                    { key: 'phi', label: 'Roll (rad)' },
                    { key: 'theta', label: 'Pitch (rad)' },
                    { key: 'psi', label: 'Yaw (rad)' },
                ],
                docs: {
                    'phi': 'roll angle in radians',
                    'theta': 'pitch angle in radians',
                    'psi': 'yaw angle in radians',
                    'Rx': 'roll rotation',
                    'Ry': 'pitch rotation',
                    'Rz': 'yaw rotation',
                    'C': 'combined 3-2-1 direction cosine matrix',
                    'v_ned': 'a body frame vector expressed in NED',
                },
            },
            {
                label: 'GPS trilateration',
                code: "sats = [15600, 7540, 20140; 18760, 2750, 18610; 17610, 14630, 13480; 19170, 610, 18390];\nrho = [21000; 21500; 22000; 21500];\nx = [0; 0; 0];\nfor it = 1:5\n    d = sats - ones(4, 1) * x';\n    r = sqrt(d(:, 1).^2 + d(:, 2).^2 + d(:, 3).^2);\n    G = -d ./ (r * ones(1, 3));\n    dx = inv(G' * G) * G' * (rho - r);\n    x = x + dx;\nend\n",
                note: 'Five Gauss-Newton steps from pseudoranges to a position fix.',
                params: [
                    { key: 'rho', label: 'Pseudoranges' },
                    { key: 'x', label: 'Initial guess' },
                ],
                docs: {
                    'sats': 'satellite positions, one per row, in km',
                    'rho': 'measured pseudoranges in km',
                    'x': 'receiver position estimate',
                    'd': 'offsets from the guess to each satellite',
                    'r': 'predicted range to each satellite',
                    'G': 'linearized geometry matrix',
                    'dx': 'least squares position correction',
                },
            },
        ],
    },
    {
        group: 'Chemical kinetics & reactors',
        items: [
            {
                label: 'Arrhenius rate sweep',
                code: 'Tlo = 300;\nThi = 900;\nT = linspace(Tlo, Thi, 25);\nA0 = 1e7;\nEa = 80000;\nRg = 8.314;\nk = A0 * exp(-Ea ./ (Rg * T));\nratio = k(25) / k(1);\n',
                note: 'Reaction rate constants over a temperature sweep with the hot-to-cold rate ratio.',
                params: [
                    { key: 'Tlo', label: 'Cold end' },
                    { key: 'Thi', label: 'Hot end' },
                    { key: 'A0', label: 'Pre-exponential factor' },
                    { key: 'Ea', label: 'Activation energy' },
                ],
                docs: {
                    'Tlo': 'coldest temperature in kelvin',
                    'Thi': 'hottest temperature in kelvin',
                    'T': 'temperature sweep in kelvin',
                    'A0': 'positive pre-exponential factor in 1/s',
                    'Ea': 'activation energy in J/mol',
                    'Rg': 'universal gas constant in J/mol/K',
                    'k': 'rate constant at each temperature',
                    'ratio': 'rate at the hottest over the coldest',
                },
            },
            {
                label: 'Batch reactor conversion',
                code: 'C0 = 2;\nk = 0.15;\ndt = 0.1;\nC = C0;\nfor n = 1:200\n    C = C - dt * k * C * C;\nend\nX = (C0 - C) / C0;\n',
                note: 'A second-order batch reaction stepped forward in time to its final conversion.',
                params: [
                    { key: 'C0', label: 'Initial concentration' },
                    { key: 'k', label: 'Rate constant' },
                    { key: 'dt', label: 'Time step' },
                ],
                docs: {
                    'C0': 'positive initial concentration in mol/L',
                    'k': 'second-order rate constant in L/mol/s',
                    'dt': 'time step in seconds',
                    'C': 'concentration stepped in time',
                    'X': 'fractional conversion at the end',
                },
            },
            {
                label: 'Catalyst deactivation',
                code: 't = linspace(0, 12, 60);\nkd = 0.25;\nk0 = 4;\nCa = 0.8;\na = exp(-kd * t);\nr = k0 * Ca * a;\nrfin = r(60);\nthalf = log(2) / kd;\n',
                note: 'Exponential catalyst decay and the effective reaction rate over a time sweep.',
                params: [
                    { key: 'kd', label: 'Deactivation constant' },
                    { key: 'k0', label: 'Fresh rate constant' },
                    { key: 'Ca', label: 'Reactant concentration' },
                ],
                docs: {
                    't': 'time samples in hours',
                    'kd': 'positive deactivation constant in 1/h',
                    'k0': 'fresh-catalyst rate constant in 1/h',
                    'Ca': 'reactant concentration in mol/L',
                    'a': 'catalyst activity at each time',
                    'r': 'effective reaction rate at each time',
                    'rfin': 'rate at the final time sample',
                    'thalf': 'activity half-life in hours',
                },
            },
            {
                label: 'CSTRs in series',
                code: 'C0 = 1.2;\nk = 0.4;\ntau = 2.5;\nC = zeros(1, 6);\nC(1) = C0;\nfor i = 1:5\n    C(i + 1) = C(i) / (1 + k * tau);\nend\nX = (C0 - C(6)) / C0;\n',
                note: 'Concentration falling through five equal stirred tanks in series.',
                params: [
                    { key: 'C0', label: 'Feed concentration' },
                    { key: 'k', label: 'Rate constant' },
                    { key: 'tau', label: 'Residence time' },
                ],
                docs: {
                    'C0': 'positive feed concentration in mol/L',
                    'k': 'positive first-order rate constant in 1/s',
                    'tau': 'positive residence time per tank in s',
                    'C': 'feed then outlet of each tank',
                    'X': 'overall conversion after five tanks',
                },
            },
        ],
    },
    {
        group: 'Civil & structural',
        items: [
            {
                label: 'Beam deflection',
                code: 'K = [2, -1, 0, 0, 0; -1, 2, -1, 0, 0; 0, -1, 2, -1, 0; 0, 0, -1, 2, -1; 0, 0, 0, -1, 2];\nq = 12;\nh = 0.5;\nEI = 4000;\nw = inv(K) * (q * h^4 / EI * ones(5, 1));\n',
                note: 'A five-node stiffness system solved for deflections under load.',
                params: [
                    { key: 'q', label: 'Load intensity' },
                    { key: 'h', label: 'Node spacing' },
                    { key: 'EI', label: 'Flexural rigidity' },
                ],
                docs: {
                    'K': 'finite difference bending stiffness',
                    'q': 'uniform load on the beam',
                    'h': 'spacing between the nodes',
                    'EI': 'flexural rigidity of the section',
                    'w': 'deflection at the five interior nodes',
                },
            },
            {
                label: 'Column buckling',
                code: 'E = 200e9;\nI = 8.1e-6;\nLv = [3, 3.5, 4, 4.5, 5];\nP = pi^2 * E * I ./ (1.0 * Lv).^2 / 1000;\n',
                note: 'Euler buckling loads swept across column lengths.',
                params: [
                    { key: 'E', label: 'Elastic modulus' },
                    { key: 'I', label: 'Second moment' },
                    { key: 'Lv', label: 'Column lengths' },
                ],
                docs: {
                    'E': 'elastic modulus in Pa',
                    'I': 'second moment of area in m^4',
                    'Lv': 'column lengths to check in meters',
                    'P': 'critical buckling load per length in kN',
                },
            },
            {
                label: 'Pipe head loss',
                code: 'L = [120, 80, 200, 150];\nD = [0.3, 0.25, 0.3, 0.2];\nv = [1.8, 2.1, 1.5, 2.4];\nf = 0.02;\nhf = f * (L ./ D) .* v.^2 / (2 * 9.81);\nworst = max(hf);\n',
                note: 'Darcy-Weisbach losses across four segments at once.',
                params: [
                    { key: 'L', label: 'Segment lengths' },
                    { key: 'D', label: 'Diameters' },
                    { key: 'v', label: 'Velocities' },
                    { key: 'f', label: 'Friction factor' },
                ],
                docs: {
                    'L': 'length of each segment in meters',
                    'D': 'inner diameter of each segment',
                    'v': 'flow velocity in each segment',
                    'f': 'Darcy friction factor',
                    'hf': 'head loss per segment in meters',
                    'worst': 'largest loss of the four',
                },
            },
            {
                label: 'Shear building sway',
                code: 'M = [200000, 0; 0, 150000];\nK = [45000000, -18000000; -18000000, 18000000];\nx = [0.05; 0.08];\nxd = [0; 0];\ndt = 0.005;\nfor k = 1:400\n    xdd = -inv(M) * (K * x);\n    xd = xd + dt * xdd;\n    x = x + dt * xd;\nend\n',
                note: 'Two stories of a building swaying through mass and stiffness.',
                params: [
                    { key: 'M', label: 'Mass matrix' },
                    { key: 'K', label: 'Stiffness matrix' },
                    { key: 'x', label: 'Initial sway' },
                ],
                docs: {
                    'M': 'floor masses',
                    'K': 'story stiffness coupling',
                    'x': 'initial displacement of each story',
                    'xd': 'story velocities',
                    'dt': 'integration step in seconds',
                    'xdd': 'acceleration from the equation of motion',
                },
            },
            {
                label: 'Truss element forces',
                code: 'c = cos(0.6);\ns = sin(0.6);\nk = 2000 * [c * c, c * s, -c * c, -c * s; c * s, s * s, -c * s, -s * s; -c * c, -c * s, c * c, c * s; -c * s, -s * s, c * s, s * s];\nu = [0; 0; 0.002; -0.001];\nf = k * u;\n',
                note: 'A bar element stiffness matrix built from its orientation.',
                params: [
                    { key: 'u', label: 'Nodal displacements' },
                ],
                docs: {
                    'c': 'cosine of the bar angle',
                    's': 'sine of the bar angle',
                    'k': 'element stiffness in global axes',
                    'u': 'displacements at both end nodes',
                    'f': 'forces the element puts on its nodes',
                },
            },
        ],
    },
    {
        group: 'Controls & estimation',
        items: [
            {
                label: 'Alpha-beta tracker',
                code: 'dt = 1.0;\nalpha = 0.85;\nbeta = 0.005;\nx = [0; 0];\nz = 1000;\nfor k = 1:10\n    xp = [x(1) + dt * x(2); x(2)];\n    rres = z - xp(1);\n    x = [xp(1) + alpha * rres; xp(2) + (beta / dt) * rres];\nend\n',
                note: 'The classic two-gain tracker smoothing noisy range measurements.',
                params: [
                    { key: 'dt', label: 'Time step' },
                    { key: 'alpha', label: 'Alpha gain' },
                    { key: 'beta', label: 'Beta gain' },
                    { key: 'z', label: 'Measured range' },
                ],
                docs: {
                    'dt': 'update interval in seconds',
                    'alpha': 'position smoothing gain',
                    'beta': 'velocity smoothing gain',
                    'x': 'tracked position and velocity',
                    'z': 'range measurement fed to every update',
                    'xp': 'prediction one interval ahead',
                    'rres': 'measurement residual',
                },
            },
            {
                label: 'Constant-velocity tracker',
                code: "dt = 0.1;\nF = [1, 0, dt, 0; 0, 1, 0, dt; 0, 0, 1, 0; 0, 0, 0, 1];\nH = [1, 0, 0, 0; 0, 1, 0, 0];\nQ = 0.01 * eye(4);\nRm = 0.5 * eye(2);\nx = zeros(4, 1);\nP = eye(4);\nz = [10; 5];\nfor k = 1:20\n    x = F * x;\n    P = F * P * F' + Q;\n    S = H * P * H' + Rm;\n    K = P * H' * inv(S);\n    x = x + K * (z - H * x);\n    P = (eye(4) - K * H) * P;\nend\n",
                note: 'Predict and correct cycles of a constant-velocity Kalman tracker.',
                params: [
                    { key: 'dt', label: 'Time step' },
                    { key: 'Q', label: 'Process noise' },
                    { key: 'Rm', label: 'Measurement noise' },
                    { key: 'z', label: 'Measurement' },
                ],
                docs: {
                    'dt': 'time between measurements',
                    'F': 'constant velocity transition',
                    'H': 'picks position out of the state',
                    'Q': 'process noise covariance',
                    'Rm': 'measurement noise covariance',
                    'x': 'position and velocity in the plane',
                    'P': 'state covariance',
                    'z': 'position fix fed to every update',
                    'S': 'innovation covariance',
                    'K': 'Kalman gain',
                },
            },
            {
                label: 'Kalman filter update',
                code: "x = zeros(4, 1);\nP = eye(4);\nH = zeros(2, 4);\nR = eye(2);\nz = zeros(2, 1);\nS = H * P * H' + R;\nK = P * H' * inv(S);\nx = x + K * (z - H * x);\nP = (eye(4) - K * H) * P;\n",
                note: 'A real filter update, tracked shape by shape with nothing flagged.',
                docs: {
                    'x': 'state estimate before the update',
                    'P': 'covariance of the state estimate',
                    'H': 'maps state into measurement space',
                    'R': 'measurement noise covariance',
                    'z': 'the incoming measurement',
                    'S': 'innovation covariance',
                    'K': 'Kalman gain',
                },
            },
            {
                label: 'PID controller',
                code: 'kp = 2.0;\nki = 0.5;\nkd = 0.1;\nsp = 1.0;\ny = 0;\ninteg = 0;\nprev = 0;\ndt = 0.05;\nfor k = 1:200\n    e = sp - y;\n    integ = integ + e * dt;\n    der = (e - prev) / dt;\n    u = kp * e + ki * integ + kd * der;\n    y = y + dt * (u - y);\n    prev = e;\nend\n',
                note: 'A full PID loop driving a first-order plant to setpoint.',
                params: [
                    { key: 'kp', label: 'P gain' },
                    { key: 'ki', label: 'I gain' },
                    { key: 'kd', label: 'D gain' },
                    { key: 'sp', label: 'Setpoint' },
                ],
                docs: {
                    'kp': 'proportional gain',
                    'ki': 'integral gain',
                    'kd': 'derivative gain',
                    'sp': 'setpoint the loop drives toward',
                    'y': 'plant output',
                    'integ': 'accumulated error',
                    'prev': 'error one step back',
                    'dt': 'loop period in seconds',
                    'e': 'tracking error',
                    'u': 'control command from the three terms',
                },
            },
            {
                label: 'State-space simulation',
                code: 'A = eye(4);\nB = zeros(4, 2);\nC = zeros(2, 4);\nx = zeros(4, 1);\nu = ones(2, 1);\nfor k = 1:10\n    x = A * x + B * u;\nend\ny = C * x;\n',
                note: 'Ten simulation steps in a loop, every shape stable throughout.',
                docs: {
                    'A': 'state transition placeholder',
                    'B': 'input map placeholder',
                    'C': 'output map placeholder',
                    'x': 'state vector',
                    'u': 'constant input',
                    'y': 'output after the ten steps',
                },
            },
        ],
    },
    {
        group: 'Data & machine learning',
        items: [
            {
                label: 'Covariance matrix',
                code: "X = zeros(200, 3);\nmu = mean(X);\nXc = X - ones(200, 1) * mu;\nC = (Xc' * Xc) / (200 - 1);\n",
                note: 'Center the data, then form the covariance matrix.',
                docs: {
                    'X': 'data matrix, one row per observation',
                    'mu': 'mean of each column',
                    'Xc': 'data with column means removed',
                    'C': 'sample covariance of the columns',
                },
            },
            {
                label: 'Least squares fit',
                code: "X = zeros(100, 3);\ny = zeros(100, 1);\nXtX = X' * X;\nbeta = inv(XtX) * X' * y;\nr = y - X * beta;\nsse = r' * r;\n",
                note: 'Normal equations for a linear fit, from data matrix to residual.',
                params: [
                    { key: 'X', label: 'Data matrix' },
                    { key: 'y', label: 'Targets' },
                ],
                docs: {
                    'X': 'design matrix, one row per sample',
                    'y': 'observed targets',
                    'XtX': 'normal equations matrix',
                    'beta': 'fitted coefficients',
                    'r': 'residuals of the fit',
                    'sse': 'sum of squared errors',
                },
            },
            {
                label: 'Logistic regression',
                code: "X = zeros(100, 3);\nyl = zeros(100, 1);\nw = zeros(3, 1);\neta = 0.1;\nfor k = 1:50\n    z = X * w;\n    prob = 1 ./ (1 + exp(-z));\n    grad = X' * (prob - yl) / 100;\n    w = w - eta * grad;\nend\n",
                note: 'Fifty gradient steps with the sigmoid applied elementwise.',
                params: [
                    { key: 'X', label: 'Data matrix' },
                    { key: 'yl', label: 'Labels' },
                    { key: 'w', label: 'Initial weights' },
                    { key: 'eta', label: 'Learning rate' },
                ],
                docs: {
                    'X': 'feature matrix, one row per sample',
                    'yl': 'labels, zero or one',
                    'w': 'weights being learned',
                    'eta': 'gradient step size',
                    'z': 'linear scores',
                    'prob': 'sigmoid probabilities',
                    'grad': 'average gradient of the loss',
                },
            },
            {
                label: 'Neural net forward pass',
                code: 'x = zeros(8, 1);\nW1 = zeros(16, 8);\nb1 = zeros(16, 1);\nh = tanh(W1 * x + b1);\nW2 = zeros(4, 16);\nb2 = zeros(4, 1);\nyhat = W2 * h + b2;\n',
                note: 'Two dense layers, weights and activations tracked end to end.',
                params: [
                    { key: 'W1', label: 'Hidden weights' },
                    { key: 'b1', label: 'Hidden bias' },
                    { key: 'W2', label: 'Output weights' },
                    { key: 'b2', label: 'Output bias' },
                ],
                docs: {
                    'x': 'input feature vector',
                    'W1': 'first layer weights',
                    'b1': 'first layer bias',
                    'h': 'hidden activations after tanh',
                    'W2': 'second layer weights',
                    'b2': 'second layer bias',
                    'yhat': 'network output',
                },
            },
        ],
    },
    {
        group: 'Electrochemistry & corrosion',
        items: [
            {
                label: 'Butler-Volmer polarization',
                code: 'Rg = 8.314;\nF = 96485;\nT = 298;\ni0 = 0.001;\nalphaA = 0.5;\nalphaC = 0.5;\neta = linspace(-0.1, 0.1, 81);\ni = i0 * (exp(alphaA * F * eta / (Rg * T)) - exp(-alphaC * F * eta / (Rg * T)));\nipk = max(i);\n',
                note: 'The Butler-Volmer net current density traced over an overpotential sweep.',
                params: [
                    { key: 'i0', label: 'Exchange current' },
                    { key: 'alphaA', label: 'Anodic coefficient' },
                    { key: 'alphaC', label: 'Cathodic coefficient' },
                    { key: 'T', label: 'Temperature' },
                ],
                docs: {
                    'Rg': 'gas constant in joules per mole kelvin',
                    'F': 'faraday constant in coulombs per mole',
                    'T': 'electrode temperature in kelvin',
                    'i0': 'exchange current density in amperes per cm2',
                    'alphaA': 'anodic charge transfer coefficient',
                    'alphaC': 'cathodic charge transfer coefficient',
                    'eta': 'overpotential sweep in volts',
                    'i': 'net current density at each overpotential',
                    'ipk': 'largest anodic current density',
                },
            },
            {
                label: 'Corrosion penetration rate',
                code: 'K = 0.00327;\nEW = 27.92;\nrho = 7.87;\nicorr = 10.^(linspace(-1, 2, 60));\ncpr = K * icorr * EW / rho;\nworst = max(cpr);\navgr = mean(cpr);\n',
                note: 'The classic penetration rate in millimeters per year across a corrosion current sweep.',
                params: [
                    { key: 'EW', label: 'Equivalent weight' },
                    { key: 'rho', label: 'Metal density' },
                    { key: 'icorr', label: 'Current density sweep' },
                ],
                docs: {
                    'K': 'unit constant for millimeters per year',
                    'EW': 'equivalent weight of the metal in grams',
                    'rho': 'metal density in grams per cubic cm',
                    'icorr': 'corrosion current density sweep',
                    'cpr': 'penetration rate in millimeters per year',
                    'worst': 'fastest penetration rate in the sweep',
                    'avgr': 'average penetration rate over the sweep',
                },
            },
            {
                label: 'Faraday electroplating',
                code: 'F = 96485;\nM = 58.69;\nn = 2;\nt = 3600;\nrho = 8.908;\nAp = 200;\nI = linspace(0.5, 5, 40);\nm = M * I * t / (n * F);\nh = 10000 * m / (rho * Ap);\n',
                note: 'Faradaic deposit mass and coating thickness computed across a plating current sweep.',
                params: [
                    { key: 't', label: 'Plating time' },
                    { key: 'Ap', label: 'Part area' },
                    { key: 'I', label: 'Current sweep' },
                    { key: 'rho', label: 'Metal density' },
                ],
                docs: {
                    'F': 'faraday constant in coulombs per mole',
                    'M': 'molar mass of the metal in grams per mole',
                    'n': 'electrons per metal ion reduced',
                    't': 'plating time in seconds',
                    'rho': 'metal density in grams per cubic cm',
                    'Ap': 'plated part area in square centimeters',
                    'I': 'applied current sweep in amperes',
                    'm': 'deposited mass at each current in grams',
                    'h': 'coating thickness in micrometers',
                },
            },
            {
                label: 'Nernst potential',
                code: 'Rg = 8.314;\nF = 96485;\nT = 298;\nn = 2;\nE0 = 0.34;\nQ = linspace(0.1, 10, 50);\nE = E0 - (Rg * T / (n * F)) * log(Q);\nEmin = min(E);\n',
                note: 'The Nernst electrode potential evaluated across a reaction quotient sweep.',
                params: [
                    { key: 'T', label: 'Temperature' },
                    { key: 'n', label: 'Electrons transferred' },
                    { key: 'E0', label: 'Standard potential' },
                    { key: 'Q', label: 'Quotient sweep' },
                ],
                docs: {
                    'Rg': 'gas constant in joules per mole kelvin',
                    'F': 'faraday constant in coulombs per mole',
                    'T': 'cell temperature in kelvin',
                    'n': 'electrons transferred in the reaction',
                    'E0': 'standard electrode potential in volts',
                    'Q': 'reaction quotient sweep',
                    'E': 'electrode potential at each quotient',
                    'Emin': 'lowest potential in the sweep',
                },
            },
        ],
    },
    {
        group: 'Energy & power',
        items: [
            {
                label: 'DC power flow',
                code: 'B = [20, -10, -10; -10, 30, -20; -10, -20, 30];\nP = [1.5; -0.5; -1.0];\ntheta = inv(B) * P;\nflow12 = 10 * (theta(1) - theta(2));\n',
                note: 'Bus angles from the susceptance matrix, then a line flow.',
                params: [
                    { key: 'B', label: 'Susceptance matrix' },
                    { key: 'P', label: 'Bus injections' },
                ],
                docs: {
                    'B': 'bus susceptance matrix',
                    'P': 'net injection at each bus',
                    'theta': 'bus voltage angles',
                    'flow12': 'flow on the line from bus 1 to bus 2',
                },
            },
            {
                label: 'Economic dispatch',
                code: 'A = [0.008, 0, -1; 0, 0.012, -1; 1, 1, 0];\nb = [-8; -6.4; 800];\nsol = inv(A) * b;\nlambda = sol(3);\n',
                note: 'Two generators and a shared marginal price from one solve.',
                params: [
                    { key: 'b', label: 'Costs and demand' },
                ],
                docs: {
                    'A': 'cost slopes and the demand balance row',
                    'b': 'cost intercepts and total demand',
                    'sol': 'generator outputs and the price',
                    'lambda': 'system marginal price',
                },
            },
            {
                label: 'Solar array output',
                code: 'G = linspace(200, 1000, 50);\nT = 25 + 0.03 * G;\nP = 0.18 * 1.6 * G .* (1 - 0.004 * (T - 25));\ntotal = sum(P);\n',
                note: 'Irradiance sweep with temperature derating, all elementwise.',
                params: [
                    { key: 'G', label: 'Irradiance sweep' },
                ],
                docs: {
                    'G': 'irradiance sweep in W/m^2',
                    'T': 'cell temperature at each level',
                    'P': 'panel power with temperature derating',
                    'total': 'output summed over the sweep',
                },
            },
            {
                label: 'Swing equation',
                code: 'Pm = 0.8;\nPmax = 1.8;\nD = 0.05;\nM = 6.5;\nx = [0.4; 0];\ndt = 0.01;\nfor k = 1:500\n    x = x + dt * [x(2); (Pm - Pmax * sin(x(1)) - D * x(2)) / M];\nend\n',
                note: 'A generator rotor angle swinging against the grid.',
                params: [
                    { key: 'Pm', label: 'Mechanical power' },
                    { key: 'Pmax', label: 'Peak transfer' },
                    { key: 'D', label: 'Damping' },
                    { key: 'M', label: 'Inertia' },
                ],
                docs: {
                    'Pm': 'mechanical input in per unit',
                    'Pmax': 'peak electrical transfer in per unit',
                    'D': 'damping coefficient',
                    'M': 'rotor inertia constant',
                    'x': 'rotor angle and speed deviation',
                    'dt': 'integration step in seconds',
                },
            },
            {
                label: 'Wind turbine power',
                code: 'v = linspace(3, 25, 45);\nP = 0.5 * 1.225 * 5027 * 0.45 * v.^3 / 1000;\ntotal = sum(P);\n',
                note: 'The cubic power curve evaluated across a wind speed sweep.',
                params: [
                    { key: 'v', label: 'Wind speed sweep' },
                ],
                docs: {
                    'v': 'wind speeds in m/s',
                    'P': 'power in kW from the cubic law',
                    'total': 'output summed over the sweep',
                },
            },
        ],
    },
    {
        group: 'Finance',
        items: [
            {
                label: 'Bond pricing',
                code: "cf = [50, 50, 50, 1050];\nt = [1, 2, 3, 4];\nr = 0.04;\nd = exp(-r * t);\nprice = cf * d';\n",
                note: 'Discounted cash flows collapsed to a price by a dot product.',
                params: [
                    { key: 'cf', label: 'Cash flows' },
                    { key: 't', label: 'Payment times' },
                    { key: 'r', label: 'Yield' },
                ],
                docs: {
                    'cf': 'coupons and the final redemption',
                    't': 'payment times in years',
                    'r': 'continuously compounded yield',
                    'd': 'discount factor for each payment',
                    'price': 'present value of the cash flows',
                },
            },
            {
                label: 'CAPM beta',
                code: "rm = [0.02, -0.01, 0.03, 0.015, -0.02, 0.025, 0.01, -0.005];\nra = [0.03, -0.015, 0.04, 0.02, -0.03, 0.035, 0.012, -0.008];\ncm = rm - mean(rm);\nca = ra - mean(ra);\nbeta = (cm * ca') / (cm * cm');\n",
                note: 'Market beta from centered return series dot products.',
                params: [
                    { key: 'rm', label: 'Market returns' },
                    { key: 'ra', label: 'Asset returns' },
                ],
                docs: {
                    'rm': 'market returns per period',
                    'ra': 'asset returns per period',
                    'cm': 'market returns, centered',
                    'ca': 'asset returns, centered',
                    'beta': 'covariance over market variance',
                },
            },
            {
                label: 'Loan amortization',
                code: 'bal = 250000;\nr = 0.045 / 12;\npmt = 1266.71;\npaid = 0;\nfor k = 1:360\n    interest = bal * r;\n    bal = bal + interest - pmt;\n    paid = paid + pmt;\nend\n',
                note: 'A mortgage balance stepped through 360 monthly payments.',
                params: [
                    { key: 'bal', label: 'Principal' },
                    { key: 'r', label: 'Monthly rate' },
                    { key: 'pmt', label: 'Monthly payment' },
                ],
                docs: {
                    'bal': 'outstanding balance',
                    'r': 'interest rate per month',
                    'pmt': 'fixed monthly payment',
                    'paid': 'total paid so far',
                    'interest': 'interest accrued this month',
                },
            },
            {
                label: 'Moving-average crossover',
                code: 't = linspace(0, 3, 120);\np = 100 + 5 * sin(2 * t) + t;\nfast = (p(3:120) + p(2:119) + p(1:118)) / 3;\nslow = (p(5:120) + p(4:119) + p(3:118) + p(2:117) + p(1:116)) / 5;\nsignal = fast(3:118) - slow;\n',
                note: 'Fast and slow averages aligned by slicing, then differenced.',
                params: [
                    { key: 'p', label: 'Price model' },
                ],
                docs: {
                    't': 'time axis for 120 sessions',
                    'p': 'price series over the axis',
                    'fast': 'three session average',
                    'slow': 'five session average',
                    'signal': 'fast minus slow, aligned',
                },
            },
            {
                label: 'Portfolio risk',
                code: "w = [0.4; 0.35; 0.25];\nSigma = [0.04, 0.01, 0; 0.01, 0.09, 0.02; 0, 0.02, 0.16];\nmu_r = [0.08; 0.11; 0.14];\nret = w' * mu_r;\nvar_p = w' * Sigma * w;\n",
                note: 'Expected return and variance from the covariance matrix.',
                params: [
                    { key: 'w', label: 'Weights' },
                    { key: 'Sigma', label: 'Covariance' },
                    { key: 'mu_r', label: 'Expected returns' },
                ],
                docs: {
                    'w': 'portfolio weights summing to one',
                    'Sigma': 'covariance of asset returns',
                    'mu_r': 'expected return of each asset',
                    'ret': 'portfolio expected return',
                    'var_p': 'portfolio variance',
                },
            },
        ],
    },
    {
        group: 'Image processing',
        items: [
            {
                label: 'Box blur',
                code: 'A = zeros(50, 50);\nB = (A(1:48, 1:48) + A(1:48, 2:49) + A(1:48, 3:50) + A(2:49, 1:48) + A(2:49, 2:49) + A(2:49, 3:50) + A(3:50, 1:48) + A(3:50, 2:49) + A(3:50, 3:50)) / 9;\n',
                note: 'A 3 by 3 mean filter built from nine shifted submatrices.',
                docs: {
                    'A': 'grayscale image',
                    'B': 'mean of each 3 by 3 neighborhood',
                },
            },
            {
                label: 'Contrast stretch',
                code: 'A = zeros(40, 60);\nlo = min(min(A));\nhi = max(max(A));\nB = (A - lo) / (hi - lo + 1);\n',
                note: 'The intensity range rescaled between its own extremes.',
                params: [
                    { key: 'A', label: 'Input image' },
                ],
                docs: {
                    'A': 'grayscale image',
                    'lo': 'darkest pixel value',
                    'hi': 'brightest pixel value',
                    'B': 'image rescaled to its own range',
                },
            },
            {
                label: 'Image downsampling',
                code: 'A = zeros(64, 64);\nsmall = A(1:2:64, 1:2:64);\ntiny = small(1:2:32, 1:2:32);\n',
                note: 'Two stepped-slice reductions, 64 to 32 to 16 pixels.',
                docs: {
                    'A': 'the full resolution image',
                    'small': 'every second pixel kept',
                    'tiny': 'quartered again',
                },
            },
            {
                label: 'Sharpening filter',
                code: 'A = zeros(50, 50);\nS = 5 * A(2:49, 2:49) - A(1:48, 2:49) - A(3:50, 2:49) - A(2:49, 1:48) - A(2:49, 3:50);\n',
                note: 'A Laplacian kernel written as shifted submatrices.',
                docs: {
                    'A': 'grayscale image',
                    'S': 'center weighted Laplacian sharpen',
                },
            },
            {
                label: 'Sobel edges',
                code: 'A = zeros(50, 50);\nGx = (A(1:48, 3:50) + 2 * A(2:49, 3:50) + A(3:50, 3:50)) - (A(1:48, 1:48) + 2 * A(2:49, 1:48) + A(3:50, 1:48));\nGy = (A(3:50, 1:48) + 2 * A(3:50, 2:49) + A(3:50, 3:50)) - (A(1:48, 1:48) + 2 * A(1:48, 2:49) + A(1:48, 3:50));\nmag = sqrt(Gx.^2 + Gy.^2);\n',
                note: 'Horizontal and vertical gradients combined into edge magnitude.',
                docs: {
                    'A': 'grayscale image',
                    'Gx': 'horizontal gradient response',
                    'Gy': 'vertical gradient response',
                    'mag': 'edge strength at each pixel',
                },
            },
        ],
    },
    {
        group: 'Life sciences',
        items: [
            {
                label: 'Leslie matrix growth',
                code: 'L = [0, 1.2, 0.8; 0.6, 0, 0; 0, 0.75, 0];\nn = [100; 60; 20];\nfor k = 1:25\n    n = L * n;\nend\ntotal = sum(n);\n',
                note: 'Age-structured population projected through a Leslie matrix.',
                params: [
                    { key: 'L', label: 'Leslie matrix' },
                    { key: 'n', label: 'Initial population' },
                ],
                docs: {
                    'L': 'births in row one, survival below',
                    'n': 'count in each age class',
                    'total': 'population after the projection',
                },
            },
            {
                label: 'Lotka-Volterra',
                code: 'y = [10; 5];\na = 1.1;\nb = 0.4;\nc = 0.1;\ndelta = 0.4;\ndt = 0.01;\nfor k = 1:1000\n    y = y + dt * [a * y(1) - b * y(1) * y(2); c * y(1) * y(2) - delta * y(2)];\nend\n',
                note: 'Predator and prey populations stepped through a thousand updates.',
                params: [
                    { key: 'a', label: 'Prey growth' },
                    { key: 'b', label: 'Predation rate' },
                    { key: 'c', label: 'Conversion rate' },
                    { key: 'delta', label: 'Predator death' },
                ],
                docs: {
                    'y': 'prey and predator counts',
                    'a': 'prey growth rate',
                    'b': 'predation rate',
                    'c': 'conversion of prey into predators',
                    'delta': 'predator death rate',
                    'dt': 'integration step',
                },
            },
            {
                label: 'Michaelis-Menten',
                code: 'S = linspace(0, 10, 100);\nVmax = 12;\nKm = 1.5;\nv = Vmax * S ./ (Km + S);\npeak = max(v);\n',
                note: 'The saturation rate curve across substrate concentrations.',
                params: [
                    { key: 'S', label: 'Substrate sweep' },
                    { key: 'Vmax', label: 'Max rate' },
                    { key: 'Km', label: 'Michaelis constant' },
                ],
                docs: {
                    'S': 'substrate concentrations',
                    'Vmax': 'maximum reaction rate',
                    'Km': 'concentration at half of Vmax',
                    'v': 'reaction rate at each concentration',
                    'peak': 'fastest rate in the sweep',
                },
            },
            {
                label: 'Pharmacokinetics',
                code: 'k10 = 0.1;\nk12 = 0.05;\nk21 = 0.03;\nA = [-(k10 + k12), k21; k12, -k21];\nx = [500; 0];\ndt = 0.1;\nfor k = 1:600\n    x = x + dt * A * x;\nend\n',
                note: 'A two-compartment drug model decaying between compartments.',
                params: [
                    { key: 'k10', label: 'Elimination rate' },
                    { key: 'k12', label: 'To tissue' },
                    { key: 'k21', label: 'To plasma' },
                    { key: 'x', label: 'Initial dose' },
                ],
                docs: {
                    'k10': 'elimination from the central compartment',
                    'k12': 'transfer into tissue',
                    'k21': 'transfer back to plasma',
                    'A': 'compartment exchange matrix',
                    'x': 'drug amount in each compartment',
                    'dt': 'time step',
                },
            },
            {
                label: 'SIR epidemic',
                code: 'y = [990; 10; 0];\nbeta = 0.3;\ngamma = 0.1;\nN = 1000;\ndt = 0.5;\nfor k = 1:200\n    inf = beta * y(1) * y(2) / N;\n    rec = gamma * y(2);\n    y = y + dt * [-inf; inf - rec; rec];\nend\n',
                note: 'Susceptible, infected, and recovered counts stepped through an outbreak.',
                params: [
                    { key: 'beta', label: 'Infection rate' },
                    { key: 'gamma', label: 'Recovery rate' },
                    { key: 'y', label: 'Initial S, I, R' },
                    { key: 'N', label: 'Population' },
                ],
                docs: {
                    'y': 'susceptible, infected, recovered',
                    'beta': 'infection rate per contact',
                    'gamma': 'recovery rate',
                    'N': 'total population',
                    'dt': 'time step in days',
                    'inf': 'new infections this step',
                    'rec': 'new recoveries this step',
                },
            },
        ],
    },
    {
        group: 'Metals & fabrication',
        items: [
            {
                label: 'Flat rolling force',
                code: 'h0 = 25;\nhf = linspace(20, 24, 9);\nR = 300;\nK = 240;\nn = 0.2;\ndraft = h0 - hf;\nL = sqrt(R * draft);\nstrain = log(h0 ./ hf);\nYf = K * strain.^n / (1 + n);\nFw = Yf .* L;\nFmax = max(Fw);\n',
                note: 'Roll separating force per unit width swept across several exit thicknesses.',
                params: [
                    { key: 'h0', label: 'Entry thickness' },
                    { key: 'hf', label: 'Exit sweep' },
                    { key: 'R', label: 'Roll radius' },
                    { key: 'K', label: 'Strength coefficient' },
                ],
                docs: {
                    'h0': 'entry thickness in mm',
                    'hf': 'positive exit thickness sweep in mm',
                    'R': 'roll radius in mm',
                    'K': 'strength coefficient in MPa',
                    'n': 'strain hardening exponent',
                    'draft': 'thickness reduction at each exit',
                    'L': 'contact length with the roll in mm',
                    'strain': 'true strain at each exit thickness',
                    'Yf': 'mean flow stress at each exit in MPa',
                    'Fw': 'roll force per unit width in N per mm',
                    'Fmax': 'largest force per width in the sweep',
                },
            },
            {
                label: 'Quench cooling',
                code: 'h = 1500;\nAs = 0.02;\nm = 1.2;\ncp = 480;\nTq = 40;\nT = 850;\ndt = 0.5;\ntau = m * cp / (h * As);\nfor k = 1:120\n    T = T + dt * (Tq - T) / tau;\nend\n',
                note: 'A lumped-capacitance steel part cooled step by step from austenitizing into a quench bath.',
                params: [
                    { key: 'h', label: 'Film coefficient' },
                    { key: 'm', label: 'Part mass' },
                    { key: 'Tq', label: 'Bath temperature' },
                    { key: 'dt', label: 'Time step' },
                ],
                docs: {
                    'h': 'positive film coefficient in W per m2 K',
                    'As': 'wetted surface area in m2',
                    'm': 'positive part mass in kg',
                    'cp': 'specific heat of steel in J per kg K',
                    'Tq': 'quench bath temperature in C',
                    'T': 'part temperature in C',
                    'dt': 'time step in seconds',
                    'tau': 'thermal time constant in seconds',
                },
            },
            {
                label: 'Springback in bending',
                code: 'Y = 350;\nE = 207000;\nt = 2;\nR = linspace(5, 100, 20);\nx = Y * R / (E * t);\nKs = 4 * x.^3 - 3 * x + 1;\nSB = 1 - Ks;\nSBmax = max(SB);\n',
                note: 'The springback ratio for sheet bending swept across a range of bend radii.',
                params: [
                    { key: 'Y', label: 'Yield strength' },
                    { key: 'E', label: 'Elastic modulus' },
                    { key: 't', label: 'Sheet thickness' },
                    { key: 'R', label: 'Radius sweep' },
                ],
                docs: {
                    'Y': 'yield strength of the sheet in MPa',
                    'E': 'positive elastic modulus in MPa',
                    't': 'positive sheet thickness in mm',
                    'R': 'bend radius sweep in mm',
                    'x': 'ratio of Y R to E t at each radius',
                    'Ks': 'springback ratio at each radius',
                    'SB': 'springback fraction at each radius',
                    'SBmax': 'largest springback in the sweep',
                },
            },
            {
                label: 'Weld peak temperature',
                code: 'T0 = 25;\nTm = 1520;\npc = 0.0044;\nt = 6;\nHnet = 1000;\ny = linspace(0, 10, 21);\ninvdT = 4.13 * pc * t * y / Hnet + 1 / (Tm - T0);\nTp = T0 + 1 ./ invdT;\nTpk = max(Tp);\n',
                note: 'The Adams peak-temperature relation solved elementwise across distances from the fusion boundary.',
                params: [
                    { key: 'y', label: 'Distance sweep' },
                    { key: 'Hnet', label: 'Heat input' },
                    { key: 't', label: 'Plate thickness' },
                    { key: 'T0', label: 'Preheat temperature' },
                ],
                docs: {
                    'T0': 'preheat temperature in C, below Tm',
                    'Tm': 'melting temperature of the metal in C',
                    'pc': 'volumetric specific heat in J per mm3 C',
                    't': 'plate thickness in mm',
                    'Hnet': 'positive net heat input in J per mm',
                    'y': 'distances from the fusion boundary in mm',
                    'invdT': 'reciprocal temperature rise at each y',
                    'Tp': 'peak temperature at each distance in C',
                    'Tpk': 'hottest peak in the sweep',
                },
            },
        ],
    },
    {
        group: 'Numerical methods',
        items: [
            {
                label: 'FFT spectrum',
                code: 't = linspace(0, 1, 128);\ns = sin(2 * pi * 8 * t) + 0.5 * sin(2 * pi * 20 * t);\nS = fft(s);\nm = abs(S);\n',
                note: 'A two-tone signal and its spectrum, lengths preserved end to end.',
                params: [
                    { key: 's', label: 'Signal model' },
                ],
                docs: {
                    't': 'one second of time, 128 samples',
                    's': 'test signal over the time axis',
                    'S': 'complex spectrum from the FFT',
                    'm': 'magnitude in each frequency bin',
                },
            },
            {
                label: 'Finite differences',
                code: 't = linspace(0, 1, 101);\nf = sin(t);\ndf = (f(2:101) - f(1:100)) / 0.01;\n',
                note: 'A forward difference from two shifted slices.',
                params: [
                    { key: 'f', label: 'Function samples' },
                ],
                docs: {
                    't': 'grid of 101 points',
                    'f': 'function sampled on the grid',
                    'df': 'forward difference derivative',
                },
            },
            {
                label: 'Forward Euler',
                code: 'A = [0, 1; -4, 0];\ny = [1; 0];\nh = 0.01;\nfor k = 1:200\n    y = y + h * (A * y);\nend\n',
                note: 'Forward Euler marching a two-state oscillator.',
                params: [
                    { key: 'A', label: 'System matrix' },
                    { key: 'y', label: 'Initial state' },
                    { key: 'h', label: 'Step size' },
                ],
                docs: {
                    'A': 'linear system being integrated',
                    'y': 'current state',
                    'h': 'step size',
                },
            },
            {
                label: 'Gradient descent',
                code: 'Q = [3, 1; 1, 2];\nb = [1; 1];\nx = zeros(2, 1);\nalpha = 0.1;\nfor k = 1:50\n    g = Q * x - b;\n    x = x - alpha * g;\nend\nr = Q * x - b;\n',
                note: 'Steepest descent on a quadratic, fifty steps with steady shapes.',
                params: [
                    { key: 'Q', label: 'Quadratic term' },
                    { key: 'b', label: 'Linear term' },
                    { key: 'alpha', label: 'Step size' },
                ],
                docs: {
                    'Q': 'symmetric positive definite quadratic',
                    'b': 'linear term of the objective',
                    'x': 'iterate, from the origin',
                    'alpha': 'fixed step size',
                    'g': 'gradient at the iterate',
                    'r': 'residual at the final iterate',
                },
            },
            {
                label: 'Heat equation',
                code: 'u = zeros(50, 1);\nu(25) = 1;\nr = 0.4;\nfor k = 1:100\n    u(2:49) = u(2:49) + r * (u(1:48) - 2 * u(2:49) + u(3:50));\nend\n',
                note: 'An explicit stencil sweep over the interior, slice lengths folded from the ranges.',
                params: [
                    { key: 'r', label: 'Diffusion number' },
                    { key: 'u(25)', label: 'Spike size' },
                ],
                docs: {
                    'u': 'temperature along the rod',
                    'u(25)': 'heat dropped in the middle',
                    'r': 'stability number, keep below 0.5',
                },
            },
            {
                label: 'KKT block assembly',
                code: "Q = [2, 0, 0; 0, 2, 0; 0, 0, 2];\nA = [1, 1, 0; 0, 1, 1];\nK = [Q, A'; A, zeros(2, 2)];\nrhs = [zeros(3, 1); ones(2, 1)];\nsol = inv(K) * rhs;\n",
                note: 'Four blocks concatenated into one saddle-point system.',
                params: [
                    { key: 'Q', label: 'Cost curvature' },
                ],
                docs: {
                    'Q': 'quadratic cost on the primal variables',
                    'A': 'two equality constraints',
                    'K': 'saddle point system in four blocks',
                    'rhs': 'gradient target and constraint values',
                    'sol': 'primal solution then multipliers',
                },
            },
            {
                label: 'Markov chain',
                code: "P = [0.9, 0.1, 0; 0.2, 0.7, 0.1; 0.1, 0.2, 0.7];\np = [1; 0; 0];\nfor k = 1:100\n    p = P' * p;\nend\n",
                note: 'A transition matrix driven to its steady state.',
                params: [
                    { key: 'P', label: 'Transition matrix' },
                    { key: 'p', label: 'Initial distribution' },
                ],
                docs: {
                    'P': 'transition probabilities, rows sum to one',
                    'p': 'probability mass on each state',
                },
            },
            {
                label: "Newton's method",
                code: 'x = [1; 1];\nfor k = 1:8\n    f = [x(1)^2 + x(2)^2 - 4; x(1) * x(2) - 1];\n    J = [2 * x(1), 2 * x(2); x(2), x(1)];\n    x = x - inv(J) * f;\nend\n',
                note: 'Newton iterations on a two-variable system, Jacobian rebuilt each pass.',
                params: [
                    { key: 'x', label: 'Initial guess' },
                ],
                docs: {
                    'x': 'current iterate',
                    'f': 'residual of the two equations',
                    'J': 'Jacobian at the iterate',
                },
            },
            {
                label: 'Polynomial interpolation',
                code: 't = [0; 1; 2; 3];\ny = [1; 2; 0; 5];\nV = [ones(4, 1), t, t.^2, t.^3];\nc = inv(V) * y;\n',
                note: 'A cubic through four points by solving the Vandermonde system.',
                params: [
                    { key: 't', label: 'Nodes' },
                    { key: 'y', label: 'Values' },
                ],
                docs: {
                    't': 'interpolation nodes',
                    'y': 'value to hit at each node',
                    'V': 'Vandermonde matrix of node powers',
                    'c': 'cubic coefficients, constant first',
                },
            },
            {
                label: 'Power iteration',
                code: "A = [2, 1; 1, 3];\nv = [1; 0];\nfor k = 1:20\n    w = A * v;\n    v = w / norm(w);\nend\nlambda = v' * A * v;\n",
                note: 'Twenty power steps converging on the dominant eigenvector.',
                params: [
                    { key: 'A', label: 'Matrix' },
                    { key: 'v', label: 'Starting vector' },
                ],
                docs: {
                    'A': 'the matrix under study',
                    'v': 'eigenvector estimate',
                    'w': 'one application of the matrix',
                    'lambda': 'Rayleigh quotient eigenvalue estimate',
                },
            },
            {
                label: 'Trapezoid rule',
                code: "h = pi / 100;\nt = linspace(0, pi, 101);\nf = sin(t);\nw = [0.5, ones(1, 99), 0.5];\nI = h * (w * f');\n",
                note: 'Composite trapezoid rule applied as a dot product.',
                params: [
                    { key: 'f', label: 'Integrand samples' },
                ],
                docs: {
                    'h': 'width of each panel',
                    't': 'sample points across the interval',
                    'f': 'integrand sampled on the grid',
                    'w': 'trapezoid weights, halved at the ends',
                    'I': 'the approximate integral',
                },
            },
        ],
    },
    {
        group: 'Petroleum & drilling',
        items: [
            {
                label: 'Exponential decline curve',
                code: 'qi = 1200;\nDi = 0.0008;\ndt = 5;\nt = linspace(0, dt * 360, 361);\nq = qi * exp(-Di * t);\nNp = sum(q) * dt;\nqmin = min(q);\n',
                note: 'An Arps exponential decline forecast with rectangle-rule cumulative production.',
                params: [
                    { key: 'qi', label: 'Initial rate' },
                    { key: 'Di', label: 'Decline rate' },
                    { key: 'dt', label: 'Sample spacing' },
                ],
                docs: {
                    'qi': 'initial rate in barrels per day',
                    'Di': 'nominal decline rate per day',
                    'dt': 'spacing between samples in days',
                    't': 'forecast times in days',
                    'q': 'declining rate in barrels per day',
                    'Np': 'cumulative production in barrels',
                    'qmin': 'lowest rate at the end of the forecast',
                },
            },
            {
                label: 'Gas material balance',
                code: 'pzi = 5300;\nG = 240;\nGp = linspace(0, 90, 7);\npz = pzi * (1 - Gp / G);\nsx = sum(Gp);\nsy = sum(pz);\nsxx = sum(Gp .* Gp);\nsxy = sum(Gp .* pz);\nA = [7, sx; sx, sxx];\nb = [sy; sxy];\ncoef = inv(A) * b;\nGest = -coef(1) / coef(2);\n',
                note: 'A pressure over z depletion line fitted by normal equations and extrapolated to gas in place.',
                params: [
                    { key: 'pzi', label: 'Initial p over z' },
                    { key: 'G', label: 'Gas in place' },
                ],
                docs: {
                    'pzi': 'initial pressure over z in psia',
                    'G': 'true gas in place in Bscf, nonzero',
                    'Gp': 'cumulative gas produced in Bscf',
                    'pz': 'pressure over z along depletion',
                    'sx': 'sum of the produced volumes',
                    'sy': 'sum of the pressure over z values',
                    'sxx': 'sum of squared produced volumes',
                    'sxy': 'sum of cross products for the fit',
                    'A': 'normal equations matrix for the line fit',
                    'b': 'right-hand side of the normal equations',
                    'coef': 'fitted intercept and slope of the line',
                    'Gest': 'extrapolated gas in place in Bscf',
                },
            },
            {
                label: 'Mud circulation hydraulics',
                code: 'q = 350;\nmu = 18;\nDh = [8.835, 8.5, 8.5];\nDp = [5, 5, 6.5];\nL = [6000, 2500, 600];\nv = q ./ (2.448 * (Dh .^ 2 - Dp .^ 2));\npf = mu * v .* L ./ (1000 * (Dh - Dp) .^ 2);\nptot = sum(pf);\nvmax = max(v);\n',
                note: 'Annular velocities and laminar friction losses summed across drillstring sections.',
                params: [
                    { key: 'q', label: 'Flow rate' },
                    { key: 'mu', label: 'Mud viscosity' },
                ],
                docs: {
                    'q': 'circulation rate in gallons per minute',
                    'mu': 'mud viscosity in centipoise',
                    'Dh': 'hole or casing diameters in inches',
                    'Dp': 'pipe outer diameters in inches',
                    'L': 'section lengths in feet',
                    'v': 'annular velocity in feet per second',
                    'pf': 'laminar friction loss per section in psi',
                    'ptot': 'total annular friction loss in psi',
                    'vmax': 'fastest annular velocity in the well',
                },
            },
            {
                label: 'Radial Darcy inflow',
                code: 'k = 0.08;\nh = 1500;\nmu = 1.2;\nre = 30000;\nrw = 10;\npe = 200;\npwf = linspace(80, 195, 24);\nJ = 2 * pi * k * h / (mu * log(re / rw));\nq = J * (pe - pwf);\nqmax = max(q);\n',
                note: 'Radial Darcy inflow rates swept across flowing bottomhole pressures.',
                params: [
                    { key: 'k', label: 'Permeability' },
                    { key: 'h', label: 'Pay thickness' },
                    { key: 'mu', label: 'Oil viscosity' },
                    { key: 'pwf', label: 'Drawdown sweep' },
                ],
                docs: {
                    'k': 'formation permeability in darcys',
                    'h': 'net pay thickness in centimeters',
                    'mu': 'oil viscosity in centipoise, nonzero',
                    're': 'external drainage radius in centimeters',
                    'rw': 'wellbore radius in centimeters',
                    'pe': 'outer boundary pressure in atmospheres',
                    'pwf': 'flowing bottomhole pressure sweep in atm',
                    'J': 'productivity index from radial darcy flow',
                    'q': 'inflow rate per drawdown in cm3 per second',
                    'qmax': 'largest rate at the deepest drawdown',
                },
            },
        ],
    },
    {
        group: 'Physics',
        items: [
            {
                label: 'Orbital motion',
                code: 'mu = 398600;\nr = [7000; 0];\nv = [0; 7.5];\ndt = 1;\nfor k = 1:120\n    a = -mu / norm(r)^3 * r;\n    v = v + dt * a;\n    r = r + dt * v;\nend\n',
                note: 'A two-body orbit integrated from gravity alone.',
                params: [
                    { key: 'r', label: 'Initial position' },
                    { key: 'v', label: 'Initial velocity' },
                    { key: 'dt', label: 'Time step' },
                ],
                docs: {
                    'mu': 'gravitational parameter in km^3/s^2',
                    'r': 'position in km',
                    'v': 'velocity in km/s',
                    'dt': 'step in seconds',
                    'a': 'gravitational acceleration',
                },
            },
            {
                label: 'Projectile with drag',
                code: 'p = [0; 0];\nv = [120; 180];\ng = [0; -9.81];\nc = 0.002;\ndt = 0.05;\nfor k = 1:100\n    a = g - c * norm(v) * v;\n    v = v + dt * a;\n    p = p + dt * v;\nend\n',
                note: 'Drag-limited projectile flight integrated step by step.',
                params: [
                    { key: 'v', label: 'Launch velocity' },
                    { key: 'c', label: 'Drag coefficient' },
                    { key: 'dt', label: 'Time step' },
                ],
                docs: {
                    'p': 'position in meters',
                    'v': 'velocity in m/s',
                    'g': 'gravitational acceleration',
                    'c': 'quadratic drag coefficient',
                    'dt': 'step in seconds',
                    'a': 'gravity plus drag',
                },
            },
            {
                label: 'Spring-mass-damper',
                code: 'm = 1;\nc = 0.4;\nk = 2;\nA = [0, 1; -k / m, -c / m];\nx = [1; 0];\ndt = 0.01;\nfor i = 1:500\n    x = x + dt * A * x;\nend\n',
                note: 'The canonical second-order system in state-space form.',
                params: [
                    { key: 'm', label: 'Mass' },
                    { key: 'c', label: 'Damping' },
                    { key: 'k', label: 'Stiffness' },
                    { key: 'x', label: 'Initial state' },
                ],
                docs: {
                    'm': 'mass',
                    'c': 'viscous damping',
                    'k': 'spring stiffness',
                    'A': 'position and velocity dynamics',
                    'x': 'displacement and velocity',
                    'dt': 'integration step',
                },
            },
            {
                label: 'Wave equation',
                code: 'u = zeros(1, 60);\nu(30) = 1;\nup = u;\nc2 = 0.25;\nfor k = 1:80\n    un = u;\n    un(2:59) = 2 * u(2:59) - up(2:59) + c2 * (u(1:58) - 2 * u(2:59) + u(3:60));\n    up = u;\n    u = un;\nend\n',
                note: 'A plucked string stepped by the leapfrog stencil.',
                params: [
                    { key: 'c2', label: 'Courant squared' },
                    { key: 'u(30)', label: 'Pluck size' },
                ],
                docs: {
                    'u': 'string displacement at 60 points',
                    'u(30)': 'pluck in the middle of the string',
                    'up': 'displacement one step back',
                    'c2': 'squared Courant number, at most one',
                    'un': 'the next displacement being assembled',
                },
            },
        ],
    },
    {
        group: 'Robotics',
        items: [
            {
                label: 'Cubic joint trajectory',
                code: 'T = 2;\nA = [1, 0, 0, 0; 0, 1, 0, 0; 1, T, T^2, T^3; 0, 1, 2 * T, 3 * T^2];\nbc = [0.2; 0; 1.5; 0];\nc = inv(A) * bc;\n',
                note: 'Boundary conditions solved for smooth joint motion coefficients.',
                params: [
                    { key: 'T', label: 'Duration' },
                    { key: 'bc', label: 'Boundary conditions' },
                ],
                docs: {
                    'T': 'motion duration in seconds',
                    'A': 'position and speed conditions at both ends',
                    'bc': 'start and end angle and speed',
                    'c': 'cubic polynomial coefficients',
                },
            },
            {
                label: 'Differential-drive odometry',
                code: 'pose = [0; 0; 0];\nv = 0.5;\nomega = 0.2;\ndt = 0.1;\nfor k = 1:100\n    pose = pose + dt * [v * cos(pose(3)); v * sin(pose(3)); omega];\nend\n',
                note: 'A robot pose integrated from wheel speed and turn rate.',
                params: [
                    { key: 'v', label: 'Forward speed' },
                    { key: 'omega', label: 'Turn rate' },
                ],
                docs: {
                    'pose': 'x, y, and heading',
                    'v': 'forward speed in m/s',
                    'omega': 'turn rate in rad/s',
                    'dt': 'odometry period in seconds',
                },
            },
            {
                label: 'Jacobian-transpose IK',
                code: "L1 = 0.5;\nL2 = 0.35;\nq = [0.3; 0.4];\ntarget = [0.6; 0.3];\nfor k = 1:100\n    p = L1 * [cos(q(1)); sin(q(1))] + L2 * [cos(q(1) + q(2)); sin(q(1) + q(2))];\n    J = [-L1 * sin(q(1)) - L2 * sin(q(1) + q(2)), -L2 * sin(q(1) + q(2)); L1 * cos(q(1)) + L2 * cos(q(1) + q(2)), L2 * cos(q(1) + q(2))];\n    e = target - p;\n    q = q + 0.5 * J' * e;\nend\n",
                note: 'The end effector pulled to a target through the transposed Jacobian.',
                params: [
                    { key: 'L1', label: 'Link 1 length' },
                    { key: 'L2', label: 'Link 2 length' },
                    { key: 'q', label: 'Initial angles' },
                    { key: 'target', label: 'Target point' },
                ],
                docs: {
                    'L1': 'first link length in meters',
                    'L2': 'second link length in meters',
                    'q': 'joint angles',
                    'target': 'goal point for the end effector',
                    'p': 'current end effector position',
                    'J': 'manipulator Jacobian',
                    'e': 'position error toward the goal',
                },
            },
            {
                label: 'PD joint control',
                code: 'M = [0.9, 0.1; 0.1, 0.4];\nKp = [25, 0; 0, 16];\nKd = [8, 0; 0, 5];\nqref = [1.0; 0.5];\nq = [0; 0];\nqd = [0; 0];\ndt = 0.002;\nfor k = 1:1000\n    tau = Kp * (qref - q) - Kd * qd;\n    qdd = inv(M) * tau;\n    qd = qd + dt * qdd;\n    q = q + dt * qd;\nend\n',
                note: 'Two joints driven to reference by proportional and derivative gains.',
                params: [
                    { key: 'Kp', label: 'P gains' },
                    { key: 'Kd', label: 'D gains' },
                    { key: 'qref', label: 'Reference angles' },
                ],
                docs: {
                    'M': 'joint space inertia matrix',
                    'Kp': 'proportional gain per joint',
                    'Kd': 'derivative gain per joint',
                    'qref': 'commanded joint angles',
                    'q': 'joint angles',
                    'qd': 'joint speeds',
                    'dt': 'control period in seconds',
                    'tau': 'commanded torques',
                    'qdd': 'joint accelerations',
                },
            },
            {
                label: 'Two-link arm kinematics',
                code: 'q1 = 0.5;\nq2 = 0.8;\nL1 = 0.5;\nL2 = 0.35;\np1 = L1 * [cos(q1); sin(q1)];\np2 = p1 + L2 * [cos(q1 + q2); sin(q1 + q2)];\nJ = [-L1 * sin(q1) - L2 * sin(q1 + q2), -L2 * sin(q1 + q2); L1 * cos(q1) + L2 * cos(q1 + q2), L2 * cos(q1 + q2)];\nvel = J * [0.1; 0.2];\n',
                note: 'End effector position and velocity through the Jacobian.',
                params: [
                    { key: 'q1', label: 'Shoulder angle' },
                    { key: 'q2', label: 'Elbow angle' },
                    { key: 'L1', label: 'Link 1 length' },
                    { key: 'L2', label: 'Link 2 length' },
                ],
                docs: {
                    'q1': 'shoulder angle in radians',
                    'q2': 'elbow angle in radians',
                    'L1': 'upper arm length in meters',
                    'L2': 'forearm length in meters',
                    'p1': 'elbow position',
                    'p2': 'end effector position',
                    'J': 'maps joint rates to tip velocity',
                    'vel': 'tip velocity for the given joint rates',
                },
            },
        ],
    },
    {
        group: 'Separations & thermodynamics',
        items: [
            {
                label: 'Antoine vapor pressure',
                code: 'T = linspace(25, 100, 100);\nA = 8.07131;\nB = 1730.63;\nC = 233.426;\nPsat = 10 .^ (A - B ./ (C + T));\npeak = max(Psat);\n',
                note: 'Water vapor pressure from the Antoine equation across a temperature sweep.',
                params: [
                    { key: 'T', label: 'Temperature sweep' },
                    { key: 'A', label: 'Antoine A' },
                    { key: 'B', label: 'Antoine B' },
                    { key: 'C', label: 'Antoine C' },
                ],
                docs: {
                    'T': 'temperatures in degrees Celsius',
                    'A': 'first Antoine coefficient for water',
                    'B': 'second Antoine coefficient for water',
                    'C': 'third Antoine coefficient for water',
                    'Psat': 'vapor pressure at each temperature in mmHg',
                    'peak': 'highest vapor pressure in the sweep',
                },
            },
            {
                label: 'Fenske minimum stages',
                code: 'xD = 0.95;\nxB = 0.05;\nalpha = linspace(1.5, 4, 100);\nsep = (xD / (1 - xD)) * ((1 - xB) / xB);\nNmin = log(sep) ./ log(alpha);\nleast = min(Nmin);\n',
                note: 'Minimum distillation stages from the Fenske equation across a relative-volatility sweep.',
                params: [
                    { key: 'xD', label: 'Distillate purity' },
                    { key: 'xB', label: 'Bottoms composition' },
                    { key: 'alpha', label: 'Volatility sweep' },
                ],
                docs: {
                    'xD': 'distillate purity between zero and one',
                    'xB': 'bottoms light fraction between zero and one',
                    'alpha': 'relative volatility sweep above one',
                    'sep': 'separation ratio of the key components',
                    'Nmin': 'minimum stages at each volatility',
                    'least': 'fewest stages at the highest volatility',
                },
            },
            {
                label: 'Kremser absorption',
                code: 'N = 6;\nA = linspace(1.1, 2.5, 100);\nf = (A .^ (N + 1) - A) ./ (A .^ (N + 1) - 1);\nslip = 1 - f;\nbest = max(f);\n',
                note: 'The absorbed solute fraction from the Kremser equation across an absorption-factor sweep.',
                params: [
                    { key: 'N', label: 'Stage count' },
                    { key: 'A', label: 'Absorption factor sweep' },
                ],
                docs: {
                    'N': 'number of equilibrium stages',
                    'A': 'absorption factor sweep above one',
                    'f': 'fraction of solute absorbed at each factor',
                    'slip': 'fraction of solute escaping the column',
                    'best': 'highest recovery in the sweep',
                },
            },
            {
                label: 'Raoult bubble point',
                code: "T = 90;\nx1 = 0.4;\nA = [6.90565, 6.95464];\nB = [1211.033, 1344.8];\nC = [220.79, 219.48];\nPsat = 10 .^ (A - B ./ (C + T));\nx = [x1, 1 - x1];\nP = x * Psat';\n",
                note: "The bubble-point pressure of a benzene and toluene liquid from Raoult's law.",
                params: [
                    { key: 'T', label: 'Temperature' },
                    { key: 'x1', label: 'Benzene mole fraction' },
                ],
                docs: {
                    'T': 'temperature in degrees Celsius',
                    'x1': 'benzene mole fraction in the liquid',
                    'A': 'first Antoine coefficient per component',
                    'B': 'second Antoine coefficient per component',
                    'C': 'third Antoine coefficient per component',
                    'Psat': 'component vapor pressures in mmHg',
                    'x': 'liquid mole fractions',
                    'P': 'bubble point total pressure in mmHg',
                },
            },
        ],
    },
    {
        group: 'Signal processing',
        items: [
            {
                label: 'Autocorrelation',
                code: "t = linspace(0, 6.2, 100);\ns = sin(3 * t);\nr0 = s * s' / 100;\nr5 = s(1:95) * s(6:100)' / 95;\nr10 = s(1:90) * s(11:100)' / 90;\nlags = [r0, r5, r10];\n",
                note: 'Three lags computed from shifted slices and stitched together.',
                params: [
                    { key: 's', label: 'Signal model' },
                ],
                docs: {
                    't': 'time axis',
                    's': 'signal built over the axis',
                    'r0': 'power at zero lag',
                    'r5': 'correlation at lag five',
                    'r10': 'correlation at lag ten',
                    'lags': 'the three lags side by side',
                },
            },
            {
                label: 'Beamforming',
                code: "X = zeros(8, 64);\nw = ones(8, 1) / 8;\ny = w' * X;\np = (y * y') / 64;\n",
                note: 'Delay-and-sum weights applied across a sensor array snapshot.',
                params: [
                    { key: 'w', label: 'Element weights' },
                ],
                docs: {
                    'X': 'snapshot of 8 sensors by 64 samples',
                    'w': 'delay and sum weights',
                    'y': 'beamformed output',
                    'p': 'average output power',
                },
            },
            {
                label: 'Decimation',
                code: 't = linspace(0, 1, 200);\ns = sin(12 * t);\nd = s(1:4:200);\n',
                note: 'Every fourth sample kept, the stepped range folded to its length.',
                params: [
                    { key: 's', label: 'Signal model' },
                ],
                docs: {
                    't': 'time axis',
                    's': 'signal to decimate',
                    'd': 'every fourth sample kept',
                },
            },
            {
                label: 'FIR filter',
                code: 't = linspace(0, 6.2, 100);\ns = sin(20 * t) + 0.3 * sin(45 * t);\ny = 0.25 * s(1:98) + 0.5 * s(2:99) + 0.25 * s(3:100);\n',
                note: 'A three-tap smoother written as shifted slice arithmetic.',
                params: [
                    { key: 's', label: 'Signal model' },
                ],
                docs: {
                    't': 'time axis',
                    's': 'input signal',
                    'y': 'three tap smoothed output',
                },
            },
        ],
    },
    {
        group: 'Option demos',
        items: [
            {
                label: 'Strict mode',
                code: "A = zeros(3, 3);\nr = A + 'error';\n",
                note: 'Turn on Strict mode to see this warning.',
                docs: {
                    'A': 'a numeric matrix',
                    'r': 'text added to a matrix, strict flags it',
                },
            },
            {
                // The fixpoint toggle changes the inferred shape of x here:
                // iterated analysis widens the growing dimension, single-pass
                // does not.
                label: 'Fixpoint loops',
                code: 'x = zeros(1, 3);\nfor i = 1:5\n    x = [x, i];\nend\ny = x * ones(4, 1);\n',
                note: 'Toggle Fixpoint to watch the shape of x change. Turn on Strict mode to see the warning.',
                docs: {
                    'x': 'row buffer grown by the loop',
                    'y': 'match depends on the settled length',
                },
            },
            {
                label: 'Shape annotations',
                code: "t = 0:0.1:6.2;\ns = sin(t);\nM = [s; cos(t)];\np = M * M';\n",
                note: 'Every inferred shape appears inline. Toggle Shape annotations to hide them.',
                params: [
                    { key: 't', label: 'Time range' },
                ],
                docs: {
                    't': 'a range with a folded length',
                    's': 'sine over the range',
                    'M': 'two rows stacked',
                    'p': 'small Gram matrix of the rows',
                },
            },
            {
                label: 'Range-length shapes',
                code: 'buf = zeros(1, 8);\nfor k = 1:4\n    if k > 2\n        buf = [buf, k];\n    end\nend\ntotal = sum(buf);\n',
                note: 'With Fixpoint on, the conditionally grown buffer gets an interval shape. Strict mode adds the reassignment warning.',
                docs: {
                    'buf': 'buffer that may grow on late passes',
                    'total': 'sum over the interval shaped buffer',
                },
            },
        ],
    },
];

export const EXAMPLES: Example[] = EXAMPLE_GROUPS.flatMap(g => g.items);

// ---------------------------------------------------------------------------
// Code generation: parameter substitution and comment injection
// ---------------------------------------------------------------------------

function escapeRegExp(s: string): string {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// The assignment line a param edits: `key = <rhs>;` at the start of a line.
// The greedy match runs to the line's last semicolon so matrix rows, which
// use `;` as a row separator, stay inside the captured right-hand side.
function paramLine(key: string): RegExp {
    return new RegExp(`^(${escapeRegExp(key)}\\s*=\\s*)(.*);[ \\t]*$`, 'm');
}

export function paramDefault(ex: Example, key: string): string {
    const m = ex.code.match(paramLine(key));
    return m ? m[2] : '';
}

export function defaultParamValues(ex: Example): { [key: string]: string } {
    const values: { [key: string]: string } = {};
    for (const p of ex.params ?? []) values[p.key] = paramDefault(ex, p.key);
    return values;
}

// Longest line that still gets its comment aligned to the common column;
// anything longer takes its comment two spaces after the code instead.
const COMMENT_ALIGN_CAP = 40;

function injectComments(ex: Example, code: string): string {
    const lines = code.split('\n');
    const targets: { idx: number; desc: string }[] = [];
    for (const [lhs, desc] of Object.entries(ex.docs ?? {})) {
        const re = new RegExp(`^\\s*(?:for\\s+)?${escapeRegExp(lhs)}\\s*=`);
        const idx = lines.findIndex(l => re.test(l) && !l.includes('%'));
        if (idx >= 0) targets.push({ idx, desc });
    }
    if (targets.length > 0) {
        const widest = Math.max(...targets.map(t => lines[t.idx].length));
        const col = Math.min(widest, COMMENT_ALIGN_CAP);
        for (const t of targets) {
            const pad = Math.max(col - lines[t.idx].length, 0) + 2;
            lines[t.idx] += ' '.repeat(pad) + '% ' + t.desc;
        }
    }
    const header = ex.note ? `% ${ex.label}: ${ex.note}` : `% ${ex.label}`;
    return `${header}\n\n${lines.join('\n')}`;
}

// Build the editor text for a template: substitute each param value into
// its assignment line (falling back to the template's own default when the
// input is blank), then inject comments if they are turned on. Always works
// from the pristine template, never from previously generated text.
export function generateCode(ex: Example, values: { [key: string]: string }, comments: boolean): string {
    let code = ex.code;
    for (const p of ex.params ?? []) {
        const raw = (values[p.key] ?? '').replace(/[\r\n]+/g, ' ').trim();
        const value = raw === '' ? paramDefault(ex, p.key) : raw;
        code = code.replace(paramLine(p.key), (_m, lhs: string) => `${lhs}${value};`);
    }
    return comments ? injectComments(ex, code) : code;
}
