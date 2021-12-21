# Method
## Chessboard
The chessboard coordinates system follows the conventions of OpenCV:
 * Right-handed coordinate system
 * x axis in direction of ascending columns
 * y axis in direction of ascending rows
 * z axis facing into the marker
## Image Coordinate System
The image coordinates system follows the conventions of OpenCV:
 * Right-handed coordinate system
 * x axis facing right
 * y axis facing down
 * z axis facing into the image
## Initial solution
The initial solution based on [A Toolbox for Easily Calibrating Omnidirectional Cameras (Davide Scaramuzza, Agostino Martinelli, Roland Siegwart)](http://rpg.ifi.uzh.ch/docs/IROS06_scaramuzza.pdf). In the following, some further details are listed to avoid ambiguity.

The initial solution is determined from a subset of all images. This is configured through the `--threshold` and `--count` arguments. The subset has to be at least `count` large and the reprojection error of each image has to be below `threshold`.

Additionally the initial solution is very sensitive to the principal point of the images (true center). If it is known it can be specified with `--principal-point`. Otherwise the package applies a brute-force heuristic by spiralling outward from the image center and trying to satisfy `count` and `threshold` at each position. This is currently done in steps of `--spiral-step` pixel in a square region with side length `--spiral-end` pixel.
### Orthonorm
To solve the scaling factor for the orthonorm vectors of the partial rotation matrix the following equation is solved. In this step only the absolute scaling factor can be determined.
$$
\begin{align*}
\mathbf{a} & = (\alpha a_1, \alpha a_2, a_3)^T \\
\mathbf{b} & = (\alpha b_1, \alpha b_2, b_3)^T \\
\mathbf{a} \cdot \mathbf{b} & = 0 \\
-(a_1 b_1 + a_2 b_2) & = a_3 b_3 \\
a_1^2 + a_2^2 + a_3^2 & = l^2 \\
b_1^2 + b_2^2 + b_3^2 & = l^2 \\
a_1^2 + a_2^2 - b_1^2 - b_2^2 & = b_3^2 - a_3^2 \\
a_1^2 + a_2^2 - b_1^2 - b_2^2 & = b_3^2 - \frac{(a_1 b_1 + a_2 b_2)^2}{b_3^2} \\
b_3^2 (a_1^2 + a_2^2 - b_1^2 - b_2^2) & = b_3^4 - (a_1 b_1 + a_2 b_2)^2 \\
b_3^4 + b_3^2 (b_1^2 + b_2^2 - a_1^2 - a_2^2 ) - (a_1 b_1 + a_2 b_2)^2 & = 0 \\
\end{align*}
$$
### Cross Product
The formula for the cross product is given only for the sake of completeness.
$$
\begin{align*}
(a^{ij}, b^{ij}, c^{ij})^T & = \begin{bmatrix}r^i_{11} &r^i_{12} & t^i_1 \\r^i_{21} &r^i_{22} & t^i_2 \\ r^i_{31} &r^i_{32} & t^i_3 \end{bmatrix} (x^{ij}, y^{ij}, 1)^T \\
(u^{ij}, v^{ij}, f(\rho^{ij}))^T \wedge (a^{ij}, b^{ij}, c^{ij})^T & = \mathbf{0} \\
v^{ij} c^{ij} - f(\rho^{ij}) b^{ij} & = 0 \\
\end{align*}
$$
$$
\begin{align*}
f(\rho^{ij}) & \left(r^i_{11} x^{ij} + r^i_{12} y^{ij} + t^i_1 \right) - & u^{ij} & \left(r^i_{31} x^{ij} + r^i_{32} y^{ij} + t^i_3 \right) & = 0 \\
u^{ij} & \left(r^i_{21} x^{ij} + r^i_{22} y^{ij} + t^i_2 \right) - & v^{ij} & \left(r^i_{11} x^{ij} + r^i_{12} y^{ij} + t^i_1 \right) & = 0 \\
v^{ij} & \left(r^i_{31} x^{ij} + r^i_{32} y^{ij} + t^i_3 \right) - & f(\rho^{ij}) & \left(r^i_{21} x^{ij} + r^i_{22} y^{ij} + t^i_2 \right) & = 0 \\
\end{align*}
$$
Note that after solving for $\mathbf{t}_3$ the complete extrinsics parameters are known. Now a check can be made if the camera lies in front or behind the plane of the chessboard marker. If the camera lies behind the chessboard it is an invalid solution and the valid solution is the one scaled by $-1$ in the **Orthonorm** step.
# Reprojection
The reprojection for the standard parametrization requires to solve a quadratic equation
$$
\begin{align*}
\rho & = \sqrt{u^2 + v^2} \\
\lambda (u, v, f(\rho))^T & = (x, y, z)^T, \; \sqrt{x^2 + y^2} = 1 \\
\rho & = \sqrt{\frac{x^2}{\lambda^2} + \frac{y^2}{\lambda^2}} \\
\lambda \rho & = \sqrt{x^2 + y^2} = 1 \\
\lambda & = \frac{1}{\rho} \\
\lambda f(\rho) & = \frac{1}{\rho} f(\rho) = z \\
f(\rho) & = \rho z \\
a_0 + (a_1 - z) \rho + \sum_{i=2}^n a_i \rho^i & = 0 \\
\text{per definition } a_1 & = 0
\end{align*}
$$
which is unwieldy for optimization. Therefore we fit a new polynom $\mathbf{b}$ so that
$$
\begin{align*}
\rho & = \sqrt{u^2 + v^2} \\
z & = a_0 + \sum_{i=2}^n a_i \rho^i \\
\mathbf{v} & = \alpha (u, v, z)^T \\
\theta & = \text{incident angle}(\mathbf{v}) = \text{arccos}\left( (0, 0, 1) \cdot \frac{v}{\left\lVert \mathbf{v}\right\rVert} \right) \\
\rho & = \sum_{i=1}^n b_i \theta^i \\
\text{per definition } b_0 & = 0
\end{align*}
$$
which can be optimized for reprojection error. Given world coordinate $(x_w, y_w, z_w)^T$ we calculate residuals as
$$
\begin{align*}
\mathbf{v}_\text{view} & = R (x_w, y_w, z_w)^T + \mathbf{t} \\
\theta & = \text{incident angle}\left(\mathbf{v}_\text{view}\right) \\
\rho & = f_\mathbf{b}(\theta) \\
r_x &= \rho \frac{x_\text{view}}{\left\lVert (x_\text{view}, y_\text{view})\right\rVert} - (x_\text{detected} - p_x)\\
r_y &= \rho \frac{y_\text{view}}{\left\lVert (x_\text{view}, y_\text{view})\right\rVert} - (y_\text{detected} - p_y)
\end{align*}
$$
and optimize parameters $R, \mathbf{t}, \mathbf{b}$ and principal point $\mathbf{p}$ via [Levenberg–Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm). Polynom $\mathbf{b}$ can easily be converted back to standard polynom $\mathbf{a}$.

The optimization is first performed on the initial subset of images and then on the full image set.
