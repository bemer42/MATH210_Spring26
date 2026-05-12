import numpy as np

def cam_creator(x, y, delta, n):
    """
    Parameters:
    x : array_like
        Degree values (e.g., spanning 0 to 180). Need not be uniformly spaced.
    y : array_like
        Cam data values corresponding to x.
    delta : float
        Degree increment used by the machine (often 1).
    n : int
        Degree of the best-fit polynomial.

    Returns:
    Cam : (k, 2) ndarray
        Column 1: x_cam = 0, delta, 2*delta, ..., x_end
        Column 2: y_cam = polynomial evaluated at x_cam
    Coeff : (n+1,) ndarray
        Polynomial coefficients (highest power first), suitable for np.polyval.
    Vel : (2,) ndarray
        [V_Start, V_End] computed using the delta increment.
    Rel_Error : float
        Relative error (%) between start and end velocities.
    """
    # Make x and y column vectors
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    m = y.size
    if m < 5:
        raise ValueError("x and y must contain at least 5 values.")
    if n < 3:
        raise ValueError("n must be at least 3 (so that n-2 >= 1).")

    # Boundary conditions vector: [y(1); y(end); 0]
    y_bou = np.array([y[0], y[-1], 0.0], dtype=float)

    # Independent variable boundary points
    x0 = x[0]
    xm = x[-1]

    # Vandermonde matrix (m x (n+1)), columns: x^n ... x^0
    A = np.vander(x, N=n + 1, increasing=False)

    # Split A into boundary and interior parts
    A_bou = A[:, [0, 1, -1]]   
    A_int = A[:, 2:-1]         

    # Define the 3x3 matrix B
    B = np.array([
        [x0**n,              x0**(n - 1),              1.0],
        [xm**n,              xm**(n - 1),              1.0],
        [(x0 + delta)**n - x0**n + (xm - delta)**n - xm**n,
         (x0 + delta)**(n - 1) - x0**(n - 1) + (xm - delta)**(n - 1) - xm**(n - 1),
         0.0]
    ], dtype=float)

    # Define the 3 x (n-2) matrix C (then flip columns left-right)
    C = np.zeros((3, n - 2), dtype=float)
    for idx, k in enumerate(range(1, n - 1)):  # k = 1..n-2
        C[0, idx] = x0**k
        C[1, idx] = xm**k
        C[2, idx] = (x0 + delta)**k - x0**k + (xm - delta)**k - xm**k
    C = np.fliplr(C)

    # Solve for interior coefficients 
    BC = np.linalg.solve(B, C)          
    By = np.linalg.solve(B, y_bou)      

    M = A_int - A_bou @ BC              
    rhs = y - A_bou @ By               

    a_int, *_ = np.linalg.lstsq(M, rhs, rcond=None) 

    # Solve for boundary coefficients
    a_bou = np.linalg.solve(B, (y_bou - C @ a_int))  

    # Official coefficient vector (length n+1): [a_bou(1:2); a_int; a_bou(end)]
    Coeff = np.concatenate([a_bou[:2], a_int, a_bou[-1:]])

    # Define curve from 0 to x(end) with step delta (inclusive within tolerance)
    x_end = x[-1]
    x_cam = np.arange(0.0, x_end + 1e-12, float(delta))
    y_cam = np.polyval(Coeff, x_cam)

    # Test velocity conditions
    V_Start = (y_cam[1] - y_cam[0]) / (x_cam[1] - x_cam[0])
    V_End = (y_cam[-1] - y_cam[-2]) / (x_cam[-1] - x_cam[-2])

    Vel = np.array([V_Start, V_End], dtype=float) 
    Vel_Error = abs((V_Start - V_End) / V_Start) * 100.0
    Point_Error = np.linalg.norm(y-np.polyval(Coeff,x)) / np.linalg.norm(y) * 100.0

    Cam = np.column_stack([x_cam, y_cam])
    return Cam, Coeff, Vel, Vel_Error, Point_Error