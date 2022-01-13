"""
Copyright 2021 Dr. David Woodburn

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
--------------------------------------------------------------------------------

The comments in this file reference variables of type "[float]".  This is
shorthand for either scalar floating-point values or Numpy ndarrays of
floating-point values.

Reference frames
----------------
    ecef_to_geodetic    (xe, ye, ze) -> phi, lam, hae
    geodetic_to_ecef    (phi, lam, hae) -> xe, ye, ze
    ecef_to_tangent     (xe, ye, ze, xe0=0, ye0=0, ze0=0, ned=True) ->
                            xt, yt, zt
    tangent_to_ecef     (xt, yt, zt, xe0=0, ye0=0, ze0=0, ned=True) ->
                            xe, ye, ze
    geodetic_to_curlin  (phi, lam, hae, phi0=0, lam0=0, hae0=0, ned=True) ->
                            xc, yc, zc
    curlin_to_geodetic  (xc, yc, zc, phi0, lam0, hae0, ned=True) ->
                            phi, lam, hae
    sv_elevation        (x_r,y_r,z_r, x_s,y_s,z_s) -> el

GPS
---
    gold10              (sv) -> ca

Plotting
--------
    plotdensity         (t,y,c="blue", bands=4)
    plotpath3           (px, py, pz, ux, uy, uz, cnt=100, scale=0.02)
    xyzlabels           (xlabel=None, ylabel=None, zlabel=None)
    plotnorm            (x,y,c="blue")

Animation
---------
    class ani_obj       (shape="rect", width=1, height=1, x=0, y=0, ang=0
                            cedge="#000", cfill="")
    class ani_frame     (objs=[], x=0, y=0, ang=0)
    ani_plot            (frames=[], file_name="ani.svg", dur=5,
                            window=[-1.618,-1,1.618,1], aspect_ratio=1.618,
                            win_pw=480):

Mathematics
-----------
    zs                  (y,x) -> xz
    dft                 (y,t) -> Y, f
    vanloan             (F, B, Q, T) -> Phi, Bd, Qd
    xcorr               (x) -> C
    avar                (y, T=1) -> va, tau
    armav               (va, tau, log_scale=False) -> vk
    sysresample         (w) -> j

Simulation
----------
    progress            (ratio)

Files
-----
    bin_read_s4         (file_name)
    bin_write_s4        (x, file_name)
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm
from scipy.optimize import leastsq
import warnings

# Constants
RAD_A = 6378137.0               # Earth's semi-major axis [m]
RAD_B = 6356752.3142            # Earth's semi-minor axis [m]
E2 = 6.6943799901413164e-3      # eccentricity squared [ND]
WIE = 7.292115e-5               # sidereal Earth rate [rad/s]
SG_GE = 9.7803253359            # Somigliana coefficient ge [m/s^2]
SG_K = 1.93185265241e-3         # Somigliana coefficient k [ND]
SG_F = 3.35281066475e-3         # Somigliana coefficient f [ND]
SG_M = 3.44978650684e-3         # Somigliana coefficient m [ND]

# ----------------
# Reference Frames
# ----------------

def ecef_to_geodetic(xe, ye, ze):
    """
    phi, lam, hae = ecef_to_geodetic(xe, ye, ze)

        Param   Description                             Type        Units
        -----   -----------                             ----        -----
        phi     geodetic latitude                       [float]     rad
        lam     longitude                               [float]     rad
        hae     height above ellipsoid                  [float]     m
        xe      ECEF x-axis position                    [float]     m
        ye      ECEF y-axis position                    [float]     m
        ze      ECEF z-axis position                    [float]     m

    This function converts an ECEF (Earth-centered, Earth-fixed) position to
    geodetic coordinates.  This follows the WGS-84 definitions (see WGS-84
    Reference System (DMA report TR 8350.2)).

    Note that inherent in solving the problem of getting the geodetic latitude
    and ellipsoidal height is finding the roots of a quartic polynomial.  While
    there are closed-form solutions to this problem (see Wikipedia), each point
    has potentially four solutions and the solutions are not numerically stable.
    Instead, this function uses the Newton-Raphson method to iteratively solve
    the problem.  The meridional position in terms of geodetic latitude, phi,
    and height above ellipsoid, hae, is

              .-        -.                 .-                 -.
              | a        |                 | a        2        |
        rho = | -- + hae | cos(phi),   z = | -- (1 - e ) + hae | sin(phi) ,
              | kp       |                 | kp                |
              '-        -'                 '-                 -'

    where e is the eccentricity of the Earth, rho is the distance from the z
    axis of the ECEF frame, and kp is

                .--------------
               /     2   2
        kp = \/ 1 - e sin (phi) .

    With these, we can get the Jacobian:

            .-          .-  2   2            -.             -.
            |           |  e a c     a        |              |
            |         s | ------- - --- - hae |           c  |
            |           |     3      kp       |              |
            |           '-  kp               -'              |
        A = |                                                | ,
            |   .-   2      2   2           2       -.       |
            |   | a e (1 - e ) s    a (1 - e )       |       |
            | c | --------------- + ---------- + hae |    s  |
            |   |         3             kp           |       |
            '-  '-      kp                          -'      -'

    where c is the cosine of phi, s is the sine of phi, and

          .-    -.   .-    -.
          | dphi |   | drho |
        A |      | = |      | .
          | dhae |   |  dz  |
          '-    -'   '-    -'

    Starting with an estimate of phi and hae, we can calculate what the
    resulting rho and z would be.  The residuals in rho and z (drho and dz,
    respectively) can be translated then to residuals in phi and hae (dphi and
    dhae, respectively) by using the inverse of the Jacobian:

        .-    -.       .-    -.
        | dphi |    -1 | drho |
        |      | = A   |      | .
        | dhae |       |  dz  |
        '-    -'       '-    -'

    We can then use these residuals to adjust our estimate of phi and hae.  Then
    we iterate again.  In testing millions of randomly generated points, three
    iterations was sufficient to reach the limit of numerical precision.

    References:
        Inertial Navigation: Theory and Implementation by David Woodburn and
            Robert Leishman
        WGS-84 Reference System (DMA report TR 8350.2)
    See also:
        geodetic_to_ecef
    Dependencies:
        import numpy as np
    Author:
        John Raquet (original)
        David Woodburn (editor)
    Version:
        2.3
    """

    # Reform the inputs to ndarrays of floats.
    x = np.asarray(xe).astype(float)
    y = np.asarray(ye).astype(float)
    z = np.asarray(ze).astype(float)

    # Initialize the height above the ellipsoid.
    hae = 0

    # Get the true radial distance from the z axis.
    rho = np.sqrt(x**2 + y**2)

    # Initialize the estimated ground latitude.
    phi = np.arctan(z/rho)  # bound to [-pi/2, pi/2]

    # Iterate to reduce residuals of the estimated closest point on the ellipse.
    for n in range(3):
        # Using the estimated ground latitude, get the cosine and sine.
        c = np.cos(phi)
        s = np.sin(phi)
        s2 = s**2

        # Get the eccentricity factor and its derivative values.
        kp2 = 1 - E2*s2
        kp = np.sqrt(kp2)
        ka = RAD_A/kp

        # Get the estimated position in the meridional plane (the plane defined
        # by the longitude and the z axis).
        rho_est = (ka + hae)*c
        z_est = (ka*(1 - E2) + hae)*s

        # Get the residuals.
        drho = rho - rho_est
        dz = z - z_est

        # Get the inverse Jacobian.
        A11 = s*(E2*RAD_A*c**2/(kp2*kp) - ka - hae)
        A12 = c
        A21 = c*(ka*(1.0 - E2) + hae
                + RAD_A*E2*(1.0 - E2)*s2/(kp2*kp))
        A22 = s
        Adet_inv = 1/(A11*A22 - A21*A12)

        # Using the inverse Jacobian, get the residuals in phi and hae.
        dphi = (A22*drho - A12*dz)*Adet_inv
        dhae = (A11*dz - A21*drho)*Adet_inv

        # Adjust the estimated ground latitude and ellipsoidal height.
        phi = phi + dphi
        hae = hae + dhae

    # Get the longitude.
    lam = np.arctan2(y, x)

    # Reduce arrays of length 1 to scalars.
    if (phi.size == 1):
        phi = phi.item()
        lam = lam.item()
        hae = hae.item()

    return phi, lam, hae

def geodetic_to_ecef(phi, lam, hae):
    """
    xe, ye, ze = geodetic_to_ecef(phi, lam, hae)

        Param   Description                             Type        Units
        -----   -----------                             ----        -----
        xe      ECEF x-axis position                    [float]     m
        ye      ECEF y-axis position                    [float]     m
        ze      ECEF z-axis position                    [float]     m
        phi     geodetic latitude                       [float]     rad
        lam     longitude                               [float]     rad
        hae     height above ellipsoid                  [float]     m

    This function converts position in geodetic coordinates to ECEF (Earth-
    centered, Earth-fixed) coordinates.  This method is direct and not an
    approximation.  This follows the WGS-84 definitions (see WGS-84 Reference
    System (DMA report TR 8350.2)).

    The distance from the z axis is

              .-  a       -.
        rho = |  --- + hae | cos(phi)
              '-  kp      -'

    where a is the semi-major radius of the earth and

                .---------------
               /     2    2
        kp = \/ 1 - e  sin (phi)
                     E

    The e sub E value is the eccentricity of the earth.  Knowing the distance
    from the z axis, we can get the x and y coordinates:

         e                       e
        x  = rho cos(lam)       y  = rho sin(lam)

    The z-axis coordinate is

         e   .-  a        2        -.
        z  = |  --- (1 - e ) + hae  | sin(phi)
             '-  kp       E        -'

    Several of these equations are admittedly not intuitively obvious.  The
    interested reader should refer to external texts for insight.

    References:
        Inertial Navigation: Theory and Implementation by David Woodburn and
            Robert Leishman
        WGS-84 Reference System (DMA report TR 8350.2)
    See also:
        ecef_to_geodetic
    Dependencies:
        import numpy as np
    Author:
        John Raquet (original)
        David Woodburn (editor)
    Version:
        2.0
    """

    # Reform the inputs to ndarrays of floats.
    phi = np.asarray(phi).astype(float)
    lam = np.asarray(lam).astype(float)
    hae = np.asarray(hae).astype(float)

    # Get the distance from the z axis.
    kp = np.sqrt(1 - E2*np.sin(phi)**2)
    rho = (RAD_A/kp + hae)*np.cos(phi)

    # Get the x, y, and z coordinates.
    xe = rho*np.cos(lam)
    ye = rho*np.sin(lam)
    ze = (RAD_A/kp*(1 - E2) + hae)*np.sin(phi)

    # Reduce arrays of length 1 to scalars.
    if (xe.size == 1):
        xe = xe.item()
        ye = ye.item()
        ze = ze.item()

    return xe, ye, ze

def ecef_to_tangent(xe, ye, ze, xe0=0, ye0=0, ze0=0, ned=True):
    """
    xt, yt, zt = ecef_to_tangent(xe, ye, ze, xe0=0, ye0=0, ze0=0, ned=True)

        Param   Description                             Type        Units
        -----   -----------                             ----        -----
        xt      local, tangent x-axis position          [float]     m
        yt      local, tangent y-axis position          [float]     m
        zt      local, tangent z-axis position          [float]     m
        xe      ECEF x-axis position                    [float]     m
        ye      ECEF y-axis position                    [float]     m
        ze      ECEF z-axis position                    [float]     m
        xe0     ECEF x-axis origin                      float       m
        ye0     ECEF y-axis origin                      float       m
        ze0     ECEF z-axis origin                      float       m
        ned     Flag to use NED or ENU orientation      bool        --

    This function converts ECEF (Earth-centered, Earth-fixed) coordinates, with
    a defined local origin, to local, tangent Cartesian North, East, Down
    (NED) or East, North, Up (ENU) coordinates.  It does this by first
    converting the ECEF origin to geodetic coordinates and using those
    coordinates to calculate a rotation matrix from the ECEF frame to the local,
    tangent Cartesian frame:

             .-                     -.
         n   |  -sp cl  -sp sl   cp  |
        R  = |    -sl     cl      0  |      NED
         e   |  -cp cl  -cp sl  -sp  |
             '-                     -'

             .-                     -.
         n   |    -sl     cl      0  |
        R  = |  -sp cl  -sp sl   cp  |      ENU
         e   |   cp cl   cp sl   sp  |
             '-                     -'

    where sp and cp are the sine and cosine of the origin latitude,
    respectively, and sl and cl are the sine and cosine of the origin longitude,
    respectively.  Then, the displacement vector of the ECEF position relative
    to the ECEF origin is rotated into the local, tangent frame:

        .-  -.      .-        -.
        | xt |    n | xe - xe0 |
        | yt | = R  | ye - ye0 |
        | zt |    e | ze - ze0 |
        '-  -'      '-        -'

    If xe0, ye0, and ze0 are not provided (or are all zeros), the first values
    of xe, ye, and ze will be used as the origin.

    See also:
        tangent_to_ecef
    Dependencies:
        import numpy as np
        ecef_to_geodetic
    Author:
        David Woodburn
    Version:
        1.0
    """

    # Reform the inputs to ndarrays of floats.
    xe = np.asarray(xe).astype(float)
    ye = np.asarray(ye).astype(float)
    ze = np.asarray(ze).astype(float)

    # Use the first point as the origin if otherwise not provided.
    if ((xe0 == 0) and (ye0 == 0) and (ze0 == 0)):
        xe0 = xe[0]
        ye0 = ye[0]
        ze0 = ze[0]

    # Get the local-level coordinates.
    phi0, lam0, hae0 = ecef_to_geodetic(xe0, ye0, ze0)

    # Get the cosines and sines of the latitude and longitude.
    cp = np.cos(phi0)
    sp = np.sin(phi0)
    cl = np.cos(lam0)
    sl = np.sin(lam0)

    # Get the displacement ECEF vector from the origin.
    dxe = xe - xe0
    dye = ye - ye0
    dze = ze - ze0

    # Get the local, tangent coordinates.
    if (ned):
        xt = -sp*cl*dxe - sp*sl*dye + cp*dze
        yt =    -sl*dxe +    cl*dye
        zt = -cp*cl*dxe - cp*sl*dye - sp*dze
    else:
        xt =    -sl*dxe +    cl*dye
        yt = -sp*cl*dxe - sp*sl*dye + cp*dze
        zt =  cp*cl*dxe + cp*sl*dye + sp*dze

    # Reduce arrays of length 1 to scalars.
    if (xt.size == 1):
        xt = xt.item()
        yt = yt.item()
        zt = zt.item()

    return xt, yt, zt

def tangent_to_ecef(xt, yt, zt, xe0, ye0, ze0, ned=True):
    """
    xe, ye, ze = tangent_to_ecef(xt, yt, zt, xe0, ye0, ze0, ned=True)

        Param   Description                             Type        Units
        -----   -----------                             ----        -----
        xe      ECEF x-axis position                    [float]     m
        ye      ECEF y-axis position                    [float]     m
        ze      ECEF z-axis position                    [float]     m
        xt      local, tangent x-axis position          [float]     m
        yt      local, tangent y-axis position          [float]     m
        zt      local, tangent z-axis position          [float]     m
        xe0     ECEF x-axis origin                      float       m
        ye0     ECEF y-axis origin                      float       m
        ze0     ECEF z-axis origin                      float       m
        ned     Flag to use NED or ENU orientation      bool        --

    This function converts local, tangent Cartesian North, East, Down (NED)
    or East, North, Up (ENU) coordinates, with a defined local origin, to ECEF
    (Earth-centered, Earth-fixed) coordinates.  It does this by first converting
    the ECEF origin to geodetic coordinates and using those coordinates to
    calculate a rotation matrix from the ECEF frame to the local, tangent
    Cartesian frame:

             .-                     -.
         e   |  -sp cl  -sl  -cp cl  |
        R  = |  -sp sl   cl  -cp sl  |      NED
         n   |    cp      0   -sp    |
             '-                     -'

             .-                     -.
         e   |   -sl  -sp cl  cp cl  |
        R  = |    cl  -sp sl  cp sl  |      ENU
         n   |     0    cp     sp    |
             '-                     -'

    where sp and cp are the sine and cosine of the origin latitude,
    respectively, and sl and cl are the sine and cosine of the origin longitude,
    respectively.  Then, the displacement vector of the ECEF position relative
    to the ECEF origin is rotated into the local, tangent frame:

        .-  -.      .-  -.   .-   -.
        | xe |    e | xt |   | xe0 |
        | ye | = R  | yt | + | ye0 |
        | ze |    n | zt |   | ze0 |
        '-  -'      '-  -'   '-   -'

    The scalars xe0, ye0, and ze0 defining the origin must be given and cannot
    be inferred.

    See also:
        ecef_to_tangent
    Dependencies:
        import numpy as np
        ecef_to_geodetic
    Author:
        David Woodburn
    Version:
        1.0
    """

    # Reform the inputs to ndarrays of floats.
    xt = np.asarray(xt).astype(float)
    yt = np.asarray(yt).astype(float)
    zt = np.asarray(zt).astype(float)

    # Get the local-level coordinates.
    phi0, lam0, hae0 = ecef_to_geodetic(xe0, ye0, ze0)

    # Get the cosines and sines of the latitude and longitude.
    cp = np.cos(phi0)
    sp = np.sin(phi0)
    cl = np.cos(lam0)
    sl = np.sin(lam0)

    # Get the local, tangent coordinates.
    if (ned):
        xe = -sp*cl*xt - sl*yt - cp*cl*zt + xe0
        ye = -sp*sl*xt + cl*yt - cp*sl*zt + ye0
        ze =     cp*xt         -    sp*zt + ze0
    else:
        xe = -sl*xt - sp*cl*yt + cp*cl*zt + xe0
        ye =  cl*xt - sp*sl*yt + cp*sl*zt + ye0
        ze =        +    cp*yt +    sp*zt + ze0

    # Reduce arrays of length 1 to scalars.
    if (xe.size == 1):
        xe = xe.item()
        ye = ye.item()
        ze = ze.item()

    return xe, ye, ze

def geodetic_to_curlin(phi, lam, hae, phi0=0, lam0=0, hae0=0, ned=True):
    """
    xc, yc, zc = geodetic_to_curlin(phi, lam, hae, phi0=0, lam0=0, hae0=0,
                                    ned=True)

        Param   Description                             Type        Units
        -----   -----------                             ----        -----
        xc      local curvilinear x-axis position       [float]     m
        yc      local curvilinear y-axis position       [float]     m
        zc      local curvilinear z-axis position       [float]     m
        phi     geodetic latitude                       [float]     rad
        lam     longitude                               [float]     rad
        hae     height above ellipsoid                  [float]     m
        phi0    geodetic latitude origin                float       rad
        lam0    longitude origin                        float       rad
        hae0    height above ellipsoid origin           float       m
        ned     Flag to use NED or ENU orientation      bool        --

    This function converts geodetic coordinates with a geodetic origin to local,
    curvilinear position in either North, East, Down (NED) or East, North, Up
    (ENU) coordinates.

    The equations are

        .-  -.   .-                                -.
        | xc |   |     (Rm + hae) (phi - phi0)      |
        | yc | = | (Rp + hae) cos(phi) (lam - lam0) |       NED
        | zc |   |           (hae0 - hae)           |
        '-  -'   '-                                -'
    or
        .-  -.   .-                                -.
        | xc |   | (Rp + hae) cos(phi) (lam - lam0) |
        | yc | = |     (Rm + hae) (phi - phi0)      |       ENU
        | zc |   |           (hae - hae0)           |
        '-  -'   '-                                -'

    where
                                     2
                             a (1 - e )                 .--------------
              a                      E                 /     2   2
        Rp = ---        Rm = ----------         kp = \/ 1 - e sin (lat)
              kp                  3                          E
                                kp

    Here, a is the semi-major axis of the Earth, e sub E is the eccentricity of
    the Earth, Rp is the parallel radius of curvature of the Earth, and Rm is
    the meridional radius of curvature of the Earth.

    If phi0, lam0, and hae0 are not provided (are left as zeros), the first
    values of phi, lam, and hae will be used as the origin.

    References:
        Titterton & Weston, "Strapdown Inertial Navigation Technology"
        https://en.wikipedia.org/wiki/Earth_radius#Meridional
        https://en.wikipedia.org/wiki/Earth_radius#Prime_vertical
    See also:
        curlin_to_geodetic
    Dependencies:
        import numpy as np
    Author:
        David Woodburn
    Version:
        1.0
    """

    # Reform the inputs to ndarrays of floats.
    phi = np.asarray(phi).astype(float)
    lam = np.asarray(lam).astype(float)
    hae = np.asarray(hae).astype(float)

    # Use the first point as the origin if otherwise not provided.
    if ((phi0 == 0) and (lam0 == 0) and (hae0 == 0)):
        phi0 = phi[0]
        lam0 = lam[0]
        hae0 = hae[0]

    # Get the parallel and meridional radii of curvature.
    kp = np.sqrt(1 - E2*np.sin(phi)**2)
    Rp = RAD_A/kp
    Rm = RAD_A*(1 - E2)/kp**3

    # Get the curvilinear coordinates.
    if (ned):   # NED
        xc = (Rm + hae)*(phi - phi0)
        yc = (Rp + hae)*np.cos(phi)*(lam - lam0)
        zc = hae0 - hae
    else:       # ENU
        xc = (Rp + hae)*np.cos(phi)*(lam - lam0)
        yc = (Rm + hae)*(phi - phi0)
        zc = hae - hae0

    # Reduce arrays of length 1 to scalars.
    if (xc.size == 1):
        xc = xc.item()
        yc = yc.item()
        zc = zc.item()

    return xc, yc, zc

def curlin_to_geodetic(xc, yc, zc, phi0, lam0, hae0, ned=True):
    """
    phi, lam, hae = curlin_to_geodetic(xc, yc, zc, phi0, lam0, hae0, ned=True)

        Param   Description                             Type        Units
        -----   -----------                             ----        -----
        phi     geodetic latitude                       [float]     rad
        lam     longitude                               [float]     rad
        hae     height above ellipsoid                  [float]     m
        xc      local curvilinear x-axis position       [float]     m
        yc      local curvilinear y-axis position       [float]     m
        zc      local curvilinear z-axis position       [float]     m
        phi0    geodetic latitude origin                float       rad
        lam0    longitude origin                        float       rad
        hae0    height above ellipsoid origin           float       m
        ned     Flag to use NED or ENU orientation      bool        --

    This function converts local, curvilinear position in either North, East,
    Down (NED) or East, North, Up (ENU) coordinates to geodetic coordinates with
    a geodetic origin.  The solution is iterative, using the Newton-Raphson
    method.

    The equations to get curvilinear coordinates from geodetic are
        .-  -.   .-                                -.
        | xc |   |     (Rm + hae) (phi - phi0)      |
        | yc | = | (Rp + hae) cos(phi) (lam - lam0) |       NED
        | zc |   |           (hae0 - hae)           |
        '-  -'   '-                                -'
    or
        .-  -.   .-                                -.
        | xc |   | (Rp + hae) cos(phi) (lam - lam0) |
        | yc | = |     (Rm + hae) (phi - phi0)      |       ENU
        | zc |   |           (hae - hae0)           |
        '-  -'   '-                                -'

    where
                                     2
                             a (1 - e )                 .--------------
              a                      E                 /     2   2
        Rp = ---        Rm = ----------         kp = \/ 1 - e sin (lat)
              kp                  3                          E
                                kp

    Here, a is the semi-major axis of the Earth, e sub E is the eccentricity of
    the Earth, Rp is the parallel radius of curvature of the Earth, and Rm is
    the meridional radius of curvature of the Earth.  Unfortunately, the reverse
    process to get geodetic coordinates from curvilinear is not as
    straightforward.  So the Newton-Raphson method is used.  Using NED as an
    example, with the above equations, we can write the differential relation as
    follows:

          .-    -.   .-      -.           .-           -.
          |  dx  |   |  dphi  |           |  A11   A12  |
        A |      | = |        |       A = |             | ,
          |  dy  |   |  dlam  |           |  A21   A22  |
          '-    -'   '-      -'           '-           -'

    where the elements of the Jacobian A are

              .-    2        -.
              |  3 e  Rm s c  |
              |     E         |
        A11 = | ------------- | (phi - phi0) + Rm + h
              '-      kp     -'

        A12 = 0

              .- .-  2    2     -.           -.
              |  |  3  s c       |            |
        A21 = |  | --------- - s | Rp - hae s | (lam - lam0)
              |  |      2        |            |
              '- '-   kp        -'           -'

        A22 = (Rp + hae) c.

    where s and c are the sine and cosine of phi, respectively.  Using the
    inverse Jacobian, we can get the residuals of phi and lam from the residuals
    of xc and yc:

                 A22 dx - A12 dy
        dphi = -------------------
                A11 A22 - A21 A12

                 A11 dy - A21 dx
        dlam = -------------------
                A11 A22 - A21 A12

    These residuals are added to the estimated phi and lam values and another
    iteration begins.

    References:
        Titterton & Weston, "Strapdown Inertial Navigation Technology"
        https://en.wikipedia.org/wiki/Earth_radius#Meridional
        https://en.wikipedia.org/wiki/Earth_radius#Prime_vertical
    See also:
        geodetic_to_curlin
    Dependencies:
        import numpy as np
    Author:
        David Woodburn
    Version:
        1.0
    """

    # Reform the inputs to ndarrays of floats.
    xc = np.asarray(xc).astype(float)
    yc = np.asarray(yc).astype(float)
    zc = np.asarray(zc).astype(float)

    # Flip the orientation if it is ENU.
    if (not ned):
        zc = zc * (-1)
        temp = xc
        xc = yc*1
        yc = temp*1

    # Define height.
    hae = hae0 - zc

    # Initialize the latitude and longitude.
    phi = phi0 + xc/(RAD_A + hae)
    lam = lam0 + yc/((RAD_A + hae)*np.cos(phi))

    # Iterate.
    for n in range(3):
        # Get the sine and cosine of latitude.
        s = np.sin(phi)
        c = np.cos(phi)

        # Get the parallel and meridional radii of curvature.
        kp2 = 1 - E2*s**2
        kp = np.sqrt(kp2)
        Rp = RAD_A/kp
        Rm = RAD_A*(1 - E2)/kp**3

        # Get the estimated xy position.
        xce = (Rm + hae)*(phi - phi0)
        yce = (Rp + hae)*c*(lam - lam0)

        # Get the residual.
        dxc = xc - xce
        dyc = yc - yce

        # Get the inverse Jacobian.
        A11 = (3*E2*Rm*s*c/kp)*(phi - phi0) + Rm + hae
        A12 = 0
        A21 = ((E2*s*c**2/kp2 - s)*Rp - hae*s)*(lam - lam0)
        A22 = (Rp + hae)*c
        Adet_inv = 1/(A11*A22 - A21*A12)

        # Using the inverse Jacobian, get the residuals in phi and lam.
        dphi = (A22*dxc - A12*dyc)*Adet_inv
        dlam = (A11*dyc - A21*dxc)*Adet_inv

        # Update the latitude and longitude.
        phi = phi + dphi
        lam = lam + dlam

    # Reduce arrays of length 1 to scalars.
    if (phi.size == 1):
        phi = phi.item()
        lam = lam.item()
        hae = hae.item()

    return phi, lam, hae

def sv_elevation(x_r, y_r, z_r, x_s, y_s, z_s):
    """
    el = sv_elevation(x_r, y_r, z_r, x_s, y_s, z_s)

        Param   Description                     Type                Units
        -----   -----------                     ----                -----
        el      Elevation angle                 float or ndarray    rad
        x_r     x-axis coordinate of receiver   float or ndarray    m
        y_r     y-axis coordinate of receiver   float or ndarray    m
        y_r     z-axis coordinate of receiver   float or ndarray    m
        x_s     x-axis coordinate of satellite  float or ndarray    m
        y_s     y-axis coordinate of satellite  float or ndarray    m
        y_s     z-axis coordinate of satellite  float or ndarray    m

    This function calculates the space vehicle (satellite) elevation angle above
    the horizon relative to the receiver.  First, it calculates the vector from
    the receiver to the space vehicle:

                                             ^   s o
        .-    -.   .-   -.   .-   -.         |    / \
        | x_rs |   | x_s |   | x_r |         |   /    \   rs
        | y_rs | = | y_s | - | y_r |         |  /       \
        | z_rs |   | z_s |   | z_r |         | /     ..--o r
        '-    -'   '-   -'   '-   -'         |/..--``
                                             o------------->

    Second, it calculates the upward vector, pointing into the sky.  This is
    actually just the <x,y,z> position vector of the receiver.  Both of these
    vectors get normalized by their respective lengths.  Third, it uses the
    definition of the dot product between those first two vectors to get the
    angle of the space vehicle above the horizon.  In calculating the upward
    vector, this function models the earth as a perfect sphere.  The error
    induced by this simplification causes at most 0.19 degrees of error.

    Dependencies:
        import numpy as np
    Authors:
        David Woodburn
    Version:
        1.0
    """

    # Reform the inputs to ndarrays of floats.
    x_r = np.asarray(x_r).astype(float)
    y_r = np.asarray(y_r).astype(float)
    z_r = np.asarray(z_r).astype(float)
    x_s = np.asarray(x_s).astype(float)
    y_s = np.asarray(y_s).astype(float)
    z_s = np.asarray(z_s).astype(float)

    # Reform one set to 2D if the other set is 2D.
    if (x_s.ndim == 2) and (x_r.ndim == 1):
        x_r = x_r.reshape(-1,1)
        y_r = y_r.reshape(-1,1)
        z_r = z_r.reshape(-1,1)

    # vector from receiver to space vehicle
    x_rs = x_s - x_r
    y_rs = y_s - y_r
    z_rs = z_s - z_r

    # norm of vector from receiver to space vehicle
    n_rs = np.sqrt(x_rs**2 + y_rs**2 + z_rs**2)

    # normalized vector
    x_rs /= n_rs
    y_rs /= n_rs
    z_rs /= n_rs

    # upward vector based on geodetic coordinates
    R = np.sqrt(x_r**2 + y_r**2 + z_r**2)
    x_up = x_r/R
    y_up = y_r/R
    z_up = z_r/R

    # elevation angle
    el = np.arcsin(x_rs*x_up + y_rs*y_up + z_rs*z_up)

    # Reduce arrays of length 1 to scalars.
    if (el.size == 1):
        el = el.item()

    return el

# ---
# GPS
# ---

def gold10(sv):
    """
    ca = gold10(sv)

        Param   Description                             Type        Units
        -----   -----------                             ----        -----
        ca      C/A code, array of 1023 1s and -1s      [float]     --
        sv      satellite number (1 to 32)              integer     --

    Given a satellite number (1 to 32) return an array of the 1023 PRN Gold code
    chip values.

    The original design of this function comes from
    https://natronics.github.io/blag/2014/gps-prn/.

    Dependencies:
        import numpy as np
    Authors:
        Nathan Bergey (original)
        David Woodburn (editor)
    Version:
        1.1
    """

    # Define the tap indices dictionary for each PRN.
    SV = {
         1: [1,5],  2: [2,6],  3: [3,7],  4: [4,8],
         5: [0,8],  6: [1,9],  7: [0,7],  8: [1,8],
         9: [2,9], 10: [1,2], 11: [2,3], 12: [4,5],
        13: [5,6], 14: [6,7], 15: [7,8], 16: [8,9],
        17: [0,3], 18: [1,4], 19: [2,5], 20: [3,6],
        21: [4,7], 22: [5,8], 23: [0,2], 24: [3,5],
        25: [4,6], 26: [5,7], 27: [6,8], 28: [7,9],
        29: [0,5], 30: [1,6], 31: [2,7], 32: [3,8],
    }

    # Define the shift function.
    def shift(register, feedback, output):
        # Calculate output.
        out = [register[i] for i in output]
        out = sum(out) % 2

        # Sum the select elements of register specified by feedback - 1 and get
        # the modulous of that sum with respect to 2.
        fb = sum([register[i] for i in feedback]) % 2

        # Shift the elements of register to the right.  The last element is
        # lost.  The second element is a duplicate of the first.
        for i in reversed(range(len(register) - 1)):
            register[i+1] = register[i]

        # Put the feedback (fb) into the first element.
        register[0] = fb

        return out

    # Initialize arrays.
    G1 = [1] * 10
    G2 = [1] * 10
    ca = np.zeros(1023)

    # Create sequence
    for i in range(1023):
        g1 = shift(G1, [2,9], [9])
        g2 = shift(G2, [1,2,5,7,8,9], SV[sv])

        # Modulo 2 add and append to the code
        ca[i] = 2*((g1 + g2) % 2) - 1

    # Return C/A code.
    return ca

# --------
# Plotting
# --------

def plotdensity(t,y,c="blue"):
    """
    Create a probability-density contour plot of y as a function of t.  The c
    input is the color specification.  For each point along t, a histogram of
    all values of the corresponding row of y is calculated.  The counts of the
    histogram are compared to 0%, 1.1%, 13.5%, and 60.5% of the maximum bin
    count.  The last three percentages were chosen to correspond to 3, 2, and 1
    standard deviation away from the mean for a normal probability distribution.
    The indices of the first and last bins to exceed those percentage points
    form the lower and upper edges of each contour.  (Actually, interpolation is
    used.)  The point is to show with darker shades where the higher density of
    y values are found.  This function does not properly handle multi-modal
    densities.

    t must be a single-dimensional array.  y must be a matrix where the number
    of rows equals the length of t.

    Dependencies:
        import numpy as np
        from matplotlib import pyplot as plt
    Authors:
        David Woodburn
    Version:
        1.3
    """

    # Check the dimensions of the inputs.
    if np.ndim(t) != 1:
        t_shape = t.shape
        if t_shape[0] > 1 and t_shape[1] > 1:
            print("plotpdf: t must be a vector!")
            return
    if np.ndim(y) != 2:
        print("plotpdf: y must be a matrix!")
        return
    y_shape = y.shape
    if y_shape[0] != len(t):
        print("plotpdf: The length of t must equal to rows of y!")
        return

    # Get the number of row and columns of y.
    rows = y_shape[0]
    cols = y_shape[1]

    # Choose the number of bins and bands.
    bands = 4
    bins = np.ceil(np.sqrt(cols)).astype(int)
    band_heights = np.array([0, 0.011, 0.135, 0.605])

    # Sort the data and add a small amount of spread.  A spread is necessary for
    # the histogram to work correctly.
    y_range = y.max() - y.min()
    dy = y_range*1e-9*(np.arange(cols) - (cols - 1)/2)
    y = np.sort(y, axis=1) + dy

    # Initialize the lower and upper edges of the bands.
    Y = np.zeros((rows,2*bands))

    # For each row of y,
    for n_row in range(rows):
        # Get this row of y.
        y_row = y[n_row,:]

        # Get the histogram of this row of the y data.
        (h,b) = np.histogram(y_row, bins)

        # Get the mid-points of the bins.
        b = (b[0:bins] + b[1:(bins+1)])/2

        # Pad the histogram with zero bins.
        db = (b[1] - b[0])*0.5
        b = np.hstack((b[0] - db, b, b[-1] + db))
        h = np.hstack((0, h, 0))

        # Normalize the bin counts.
        h_max = np.max(h)
        h = h/h_max

        # For this row of y, define the lower and upper edges of the bands.
        Y[n_row, 0] = b[0]
        Y[n_row, 1] = b[-1]
        for n_band in range(1, bands):
            # Get the index before the first value greater than the threshold
            # and the last index of the last value greater than the threshold.
            z = h - band_heights[n_band]
            n = np.nonzero(z >= 0)[0]
            n_a = n[0] - 1
            n_b = n[-1]

            # Interpolate bin locations to find the correct y values of the
            # bands.
            b_a = b[n_a] + (b[n_a+1] - b[n_a])*(0 - z[n_a])/(z[n_a+1] - z[n_a])
            b_b = b[n_b] + (b[n_b+1] - b[n_b])*(0 - z[n_b])/(z[n_b+1] - z[n_b])

            # Store the interpolated bin values.
            Y[n_row,(n_band*2)] = b_a
            Y[n_row,(n_band*2+1)] = b_b

    # Plot each band as an overlapping, filled area with 20% opacity.
    for n_band in range(0,bands):
        n_col = (n_band*2)
        plt.fill_between(t.flatten(),Y[:,n_col],Y[:,n_col+1],
                alpha=0.2, color=c)

def plotpath3(px, py, pz, ux, uy, uz, cnt=100, scale=0.02):
    """
    Plot a 3D curve from (px,py,pz) and plot on top of that vectors from
    (ux,uy,uz) at regular intervals of (px,py,pz).  The total number of vectors
    plotted can be controlled with the cnt parameter (default 100).  The mean
    length of the vectors relative to the overall size of the plot box is set by
    the scale parameter (default 0.02).

    Dependencies:
        import numpy as np
        from matplotlib import pyplot as plt
    Authors:
        David Woodburn
    Version:
        1.3
    """

    # Check the lengths
    Npx = len(px);      Npy = len(py);      Npz = len(pz)
    if (Npx != Npy) or (Npy != Npz):
        print("plotpath3: px, py, and pz must be the same lengths!")
        return
    Np = Npx
    Nux = len(ux);      Nuy = len(uy);      Nuz = len(uz)
    if (Nux != Nuy) or (Nuy != Nuz):
        print("plotpath3: ux, uy, and uz must be the same lengths!")
        return
    Nu = Nux
    if (Np != Nu):
        print("plotpath3: (px,py,pz) and (ux,uy,uz) must be the same lengths!")
        return
    N = Np

    # Get the vector scaling factor.
    u = np.mean(np.sqrt(ux**2 + uy**2 + uz**2))
    X = np.max(px) - np.min(px)
    Y = np.max(py) - np.min(py)
    Z = np.max(pz) - np.min(pz)
    P = np.sqrt(X**2 + Y**2 + Z**2)
    scaling = (P/u)*scale

    # Get the interval step.
    if cnt > N:
        cnt = N
    step = np.ceil(N/cnt).astype(int)

    # Plot the path and vectors.
    plt.gca(projection='3d')
    plt.plot(px, py, pz)
    plt.quiver(px[::step], py[::step], pz[::step],
            ux[::step]*scaling, uy[::step]*scaling, uz[::step]*scaling,
            color="orange", arrow_length_ratio=0, linewidth=1)

def xyzlabels(xlabel=None, ylabel=None, zlabel=None):
    """
    Add x, y, and z axis labels to a 3D plot.

    Dependencies:
        from matplotlib import pyplot as plt
    Authors:
        David Woodburn
    Version:
        1.0
    """

    ax = plt.gca()
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    if zlabel != None:
        ax.set_zlabel(zlabel)

def plotnorm(x, y, c="blue"):
    """
    This function will plot the mean and the mean +/- one standard deviation of
    the y input versus x.

    Dependencies:
        import numpy as np
        from matplotlib import pyplot as plt
    Authors:
        David Woodburn
    Version:
        1.0
    """

    # Make sure y is 2D.
    y_shape = y.shape
    if len(y_shape) != 2:
        print("plotnorm: y must be 2D!")
        return

    # Get the statistics.
    mu = np.mean(y, axis=1)
    sd = np.std(y, axis=1)

    # Plot.
    plt.plot(x, mu, color=c)
    plt.plot(x, mu + sd, color=c, linestyle='--')
    plt.plot(x, mu - sd, color=c, linestyle='--')

# ---------
# Animation
# ---------

class ani_obj:
    """
    This class simply stores the specifications of a single svg animation
    object.  The specifications are

        shape   shape of the object to create
        width   width of the object in user dimensions
        height  height of the object in user dimensions
        x       x center of object relative to frame in user dimensions
        y       y center of object relative to frame in user dimensions
        ang     rotation angle in degrees of the object relative to the frame
        cedge   color of the edge lines as a hex RGB string (e.g. "#fff")
        cfill   color of the shape fill as a hex RGB string (e.g. "#fff")

    The shape input must be string, one of the following:

        rect    rectangle
        line    line
        circ    circle or ellipse
        itri    isosceles triangle
        rtri    right triangle with the right angle on the right
        ltri    right triangle with the right angle on the left

    Once an object or many objects have been defined, they can be assembled as a
    Python list into a frame using ani_frame.  Then, many frames can be
    assembled as a Python list and sent to ani_plot to create an animation file.

    Dependencies:
        (none)
    Authors:
        David Woodburn
    Version:
        1.0
    """

    def __init__(self, shape="rect", width=1, height=1, x=0, y=0, ang=0,
            cedge="#000", cfill=""):
        # Check that shape is valid.
        if shape not in ["rect", "line", "circ", "itri", "rtri", "ltri"]:
            print(f"ani_obj: did not recognize shape \"{shape}\"")
            self.shape = "rect"

        # Make sure width, height, x, y, and ang are numbers.
        if (type(width) != int) and (type(width) != float):
            print("ani_obj: width must be a number!")
            return
        if (type(height) != int) and (type(height) != float):
            print("ani_obj: height must be a number!")
            return
        if (type(x) != int) and (type(x) != float):
            print("ani_obj: x must be a number!")
            return
        if (type(y) != int) and (type(y) != float):
            print("ani_obj: y must be a number!")
            return
        if (type(ang) != int) and (type(ang) != float):
            print("ani_obj: ang must be a number!")
            return

        # Assign the inputs to the object.
        self.shape = shape
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.ang = ang
        self.cedge = cedge
        self.cfill = cfill

class ani_frame:
    """
    This class simply stores the specifications of an animation frame.  A frame
    is a group of animation objects which can be moved together.  The
    specifications are

        objs    the list of animation objects
        x       the scalar or ndarray of x-axis position values
        y       the scalar or ndarray of y-axis position values
        ang     the scalar or ndarray of rotation angle values in degrees

    Dependencies:
        (none)
    Authors:
        David Woodburn
    Version:
        1.0
    """

    def __init__(self, objs=[], x=0, y=0, ang=0):
        # Make sure objs is of type ani_obj or a list of such.
        if type(objs) == ani_obj:
            objs = [objs]
        elif (type(objs) != list) or (type(objs[0]) != ani_obj):
            print("ani_frame: objs must be of type ani_obj "
                    "or a list of such!")
            return

        # Make sure x and y and ang are NumPy arrays.
        if (type(x) == int) or (type(x) == float):
            x = np.array([x])
        elif (type(x) != np.ndarray):
            print("ani_frame: x must be a number or a NumPy array!")
            return
        if (type(y) == int) or (type(y) == float):
            y = np.array([y])
        elif (type(y) != np.ndarray):
            print("ani_frame: y must be a number or a NumPy array!")
            return
        if (type(ang) == int) or (type(ang) == float):
            ang = np.array([ang])
        elif (type(ang) != np.ndarray):
            print("ani_frame: ang must be a number or a NumPy array!")
            return

        # Make sure x and y are the same lengths.
        if len(x) != len(y):
            if len(x) == 1:
                x = x * np.ones(len(y))
            elif len(y) == 1:
                y = y * np.ones(len(x))
            else:
                print("ani_frame: x and y must be the same lengths "
                        "or one must be a scalar!")

        # Unwrap the angle.  This is necessary to prevent glitchy behavior at
        # the zero-crossing.
        ang = np.unwrap(ang)

        # Assign the inputs to the object.
        self.objs = objs
        self.x = x
        self.y = y
        self.ang = ang

def ani_plot(frames=[], file_name="ani.svg", dur=5, window=[-1.618,-1,1.618,1],
        aspect_ratio=1.618, win_pw=480):
    """
    This function creates an svg animation file.  It takes the following
    parameters

        frames          a list of ani_frame frames
        file_name       the name of the animation file
        dur             the total duration of the animation in seconds
        window          the window of visibility specified by the bottom left
                            and top right corners, in user dimensions
        aspect_ratio    the aspect ratio (width/height) of the animation in
                            pixels
        min_pw          the animation width in pixels

    Dependencies:
        import numpy as np
    Authors:
        David Woodburn
    Version:
        1.0
    """

    # Make sure frames is of type ani_frame or a list of such.
    if type(frames) == ani_frame:
        frames = [frames]
    elif (type(frames) != list) or (type(frames[0]) != ani_frame):
        print("ani_plot: frames must be of type ani_frame "
                "or a list of such!")
        return

    # Open the file.
    fid = open(file_name, "w")
    #if IOError:
    #    print(f"ani_plot: Could not open file {file_name}!")
    #    return

    # Get the bottom left corner of the window and the width and height of the
    # window.
    win_x = window[0]
    win_y = window[1]
    win_w = window[2] - window[0]
    win_h = window[3] - window[1]

    # Get the window pixel height.
    win_ph = int(round(win_pw / aspect_ratio / 2.0) * 2)

    # Write the header.
    fid.write(f"<svg viewBox=\"-{win_pw/2:.0f} -{win_ph/2:.0f}")
    fid.write(f" {win_pw:.0f} {win_ph:.0f}\"\n")
    fid.write("  xmlns=\"http://www.w3.org/2000/svg\">\n")

    # For each frame,
    for n_frame in range(len(frames)):
        # Parse out the frame properties.
        frame = frames[n_frame]
        objs = frame.objs
        frame_px = (frame.x - win_x)/win_w * win_pw - win_pw/2
        frame_py = win_ph/2 - (frame.y - win_y)/win_h * win_ph
        frame_ang = -frame.ang / np.pi * 180.0

        # Open the group.
        fid.write("\n  <g")
        if (len(frame_px) == 1) or (len(frame_ang) == 1):
            fid.write(" transform=\"")
            if (len(frame_px) == 1) and (frame_px[0] != 0 or frame_py[0] != 0):
                fid.write(f" translate({frame_px[0]:.2f} "
                        f"{frame_py[0]:.1f})")
            if (len(frame_ang) == 1) and (frame_ang[0] != 0):
                fid.write(f" rotate({frame_ang[0]:.2f})")
            fid.write("\"")
        fid.write(">\n")

        # For each object,
        for n_obj in range(len(frame.objs)):
            # Parse out the object properties.
            obj = frame.objs[n_obj]
            shape = obj.shape
            obj_pw = obj.width / win_w * win_pw
            obj_ph = obj.height / win_h * win_ph
            obj_px = obj.x / win_w * win_pw
            obj_py = -obj.y / win_h * win_ph
            obj_ang = -obj.ang / np.pi * 180.0
            cedge = obj.cedge
            cfill = obj.cfill

            # Define the object.
            if shape == "line":
                fid.write("    <line\n")
                fid.write("     ")
                fid.write(f" x1=\"{obj_px - obj_pw*0.5:.1f}\"")
                fid.write(f" y1=\"{obj_py + obj_ph*0.5:.1f}\"\n")
                fid.write("     ")
                fid.write(f" x2=\"{obj_px + obj_pw*0.5:.1f}\"")
                fid.write(f" y2=\"{obj_py - obj_ph*0.5:.1f}\"\n")
            elif shape == "rect":
                fid.write("    <rect\n")
                fid.write("     ")
                fid.write(f" x=\"{obj_px - obj_pw*0.5:.1f}\"")
                fid.write(f" y=\"{obj_py - obj_ph*0.5:.1f}\"\n")
                fid.write("     ")
                fid.write(f" width=\"{obj_pw:.1f}\"")
                fid.write(f" height=\"{obj_ph:.1f}\"\n")
            elif shape == "circ":
                fid.write("    <ellipse\n")
                fid.write("     ")
                fid.write(f" cx=\"{obj_px:.1f}\"")
                fid.write(f" cy=\"{obj_py:.1f}\"\n")
                fid.write("     ")
                fid.write(f" rx=\"{obj_pw*0.5:.1f}\"")
                fid.write(f" ry=\"{obj_ph*0.5:.1f}\"\n")
            elif shape == "itri":
                fid.write("    <polygon\n")
                fid.write("      points=\"")
                fid.write(f"{obj_px - obj_pw*0.5:.1f},"
                        f"{obj_py + obj_ph*0.5:.1f}")
                fid.write(f" {obj_px + obj_pw*0.5:.1f},"
                        f"{obj_py + obj_ph*0.5:.1f}")
                fid.write(f" {obj_px:.1f},"
                        f"{obj_py - obj_ph*0.5:.1f}\"\n")
            elif shape == "rtri":
                fid.write("    <polygon\n")
                fid.write("      points=\"")
                fid.write(f"{obj_px - obj_pw*0.5:.1f},"
                        f"{obj_py + obj_ph*0.5:.1f}")
                fid.write(f" {obj_px + obj_pw*0.5:.1f},"
                        f"{obj_py + obj_ph*0.5:.1f}")
                fid.write(f" {obj_px + obj_pw*0.5:.1f},"
                        f"{obj_py - obj_ph*0.5:.1f}\"\n")
            elif shape == "ltri":
                fid.write("    <polygon\n")
                fid.write("      points=\"")
                fid.write(f"{obj_px - obj_pw*0.5:.1f},"
                        f"{obj_py + obj_ph*0.5:.1f}")
                fid.write(f" {obj_px + obj_pw*0.5:.1f},"
                        f"{obj_py + obj_ph*0.5:.1f}")
                fid.write(f" {obj_px - obj_pw*0.5:.1f},"
                        f"{obj_py - obj_ph*0.5:.1f}\"\n")
            if obj_ang != 0:
                fid.write(f"      transform=\"rotate({obj_ang:.1f}"
                        f" {obj_px:.1f} {obj_py:.1f})\"\n")
            if cfill == "":
                fid.write(f"      fill-opacity=\"0\"")
            else:
                fid.write(f"      fill=\"{cfill}\"")
            if cedge != "":
                fid.write(f" stroke=\"{cedge}\"")
                fid.write(f" stroke-width=\"1\"")
            fid.write("/>\n")

        # Move the group.
        if len(frame_px) > 1:
            # Open the transform.
            fid.write( "\n    <animateTransform\n")
            fid.write( "      attributeType=\"XML\"\n")
            fid.write( "      attributeName=\"transform\"\n")
            fid.write( "      type=\"translate\"\n")
            fid.write( "      additive=\"sum\"\n")
            fid.write( "      repeatCount=\"indefinite\"\n")
            fid.write(f"      dur=\"{dur:.2f}\"\n")

            # Write the transform values.
            fid.write(f"      values=\"")
            for n_point in range(len(frame_px) - 1):
                if (n_point % 5 == 0):
                    fid.write("\n       ")
                fid.write(f" {frame_px[n_point]:.1f},"
                        f"{frame_py[n_point]:.1f};")
            if ((len(frame_px) - 1) % 5 == 0):
                fid.write("\n       ")
            fid.write(f" {frame_px[-1]:.1f},{frame_py[-1]:.1f}\"/>\n")

        # Rotate the group.
        if len(frame_ang) > 1:
            # Open the transform.
            fid.write( "\n    <animateTransform\n")
            fid.write( "      attributeType=\"XML\"\n")
            fid.write( "      attributeName=\"transform\"\n")
            fid.write( "      type=\"rotate\"\n")
            fid.write( "      additive=\"sum\"\n")
            fid.write( "      repeatCount=\"indefinite\"\n")
            fid.write(f"      dur=\"{dur:.2f}\"\n")

            # Write the transform values.
            fid.write(f"      values=\"")
            for n_point in range(len(frame_ang) - 1):
                if (n_point % 5 == 0):
                    fid.write("\n       ")
                fid.write(f" {frame_ang[n_point]:.2f};")
            if ((len(frame_ang) - 1) % 5 == 0):
                fid.write("\n       ")
            fid.write(f" {frame_ang[-1]:.2f}\"/>\n")

        # group closing
        fid.write("  </g>\n")

    # Write the closing.
    fid.write("</svg>\n")

    # Close the file.
    fid.close()

# -----------
# Mathematics
# -----------

def zs(y,x):
    """
    This function finds the zero-crossings of y and returns the linearly
    interpolated corresponding values of x.  This function exists because the
    `interp` function in NumPy does not handle non-monotonically changing values
    of x.

    Dependencies:
        import numpy as np
    Authors:
        David Woodburn
    Version:
        1.0
    """

    # Get indexes of y where sign changes.
    N = len(x)
    isz = np.append((np.sign(y[0:(N-1)]) != np.sign(y[1:N])), False)
    nn = np.where(isz)[0]

    # Interpolate to get the corresponding values of x.
    xz = x[nn] - (x[nn+1] - x[nn])/(y[nn+1] - y[nn])*y[nn]
    return xz

def dft(y,t):
    """
    This function generates the single-sided Fourier transform of y(t) returning
    Y(f) in the order Y,f.  All arrays are expected to be 1D arrays.  The Y
    array is the complex-valued Fourier transform of y.  The f array is the
    frequency array in Hz from 0 to the Nyquist limit.  This function is
    designed such that the function

        y(t) = 3 + sin(2*pi*t) + 7*cos(2*pi*10*t)

    will result in the following frequency components:

        Freq  | Magnitude
        ----- | ---------
        0 Hz  | 3
        1 Hz  | 1
        10 Hz | 7

    Dependencies:
        import numpy as np
    Authors:
        David Woodburn
    Version:
        1.0
    """

    # Get the scaled Fourier transform of y.
    Nt = len(t)
    Y = np.fft.fft(y)/Nt

    # Crop the Fourier transform to the first half of the data (below the
    # Nyquist limit) and finish the scaling.  The DC component should not be
    # doubled.
    Nt_h = np.floor(Nt/2).astype(int) + 1
    Y = Y[:Nt_h]*2
    Y[0] /= 2

    # Build the frequency array.
    T = np.mean(np.diff(t))
    df = 1/((Nt-1)*T)
    f = np.arange(Nt_h)*df

    return (Y,f)

def vanloan(F, B, Q, T):
    """
    Apply the Van Loan method to the matrices F, B, and Q.  T is the sampling
    period in seconds.  The Van Loan method is one way of discretizing the
    matrices of a state-space system.  Suppose that you have the following
    state-space system:
        .                 .--
        x = F x + B u + \/ Q  w

        y = C x + D u + R v

    where x is the state vector, u is the input vector, and w is a white,
    Gaussian noise vector with means of zero and variances of one.  Then, to get
    the discrete form of this equation, we would need to find Phi, Bd, and Qd
    such that
                             .--
        x = Phi x + Bd u + \/ Qd w

        y = C x + D u + Rd v

    Rd is simply R/T.  C and D are unaffected by the discretization process.  We
    can find Phi and Qd by doing the following:
            .-      -.                    .-          -.
            | -F  Q  |                    |  M11  M12  |
        L = |        |    M = expm(L T) = |            |
            |  0  F' |                    |  M21  M22  |
            '-      -'                    '-          -'
        Phi = M22'        Qd = Phi M12

    Note that F must be square and Q must have the same size as F.  To find Bd,
    we do the following:
            .-      -.                    .-         -.
            |  F  B  |                    |  Phi  Bd  |
        G = |        |    H = expm(G T) = |           |
            |  0  0  |                    |   0   I   |
            '-      -'                    '-         -'
    Note that B must have the same number of rows as F, but need not have the
    same number of columns.  The first-order (Euler method) approximation to
    these is as follows:

        Phi = I + F T
        Bd  = B T
        Qd  = Q T

    For Bd to be calculated, F and B must have the same number of rows.  For Qd
    to be calculated, F and Q must have the same shape.  If these conditions are
    not met, the corresponding output will be a zero 1x1 array.

    We can also express Phi and Bd in terms of their infinite series:

                         1   2  2    1   3  3
        Phi = I + F T + --- F  T  + --- F  T  + ...
                         2!          3!

                    1       2    1   2    3    1   3    4
        Bd = B T + --- F B T  + --- F  B T  + --- F  B T  + ...
                    2!           3!            4!

    The Van Loan method is named after Charles Van Loan.

    References:
        C. Van Loan, "Computing Integrals Involving the Matrix Exponential,"
            1976.
        Brown, R. and Phil Hwang. "Introduction to Random Signals and Applied
            Kalman Filtering (4th ed.)" (2012).
        https://en.wikipedia.org/wiki/Discretization
    Dependencies:
        import numpy as np
        from scipy.linalg import expm
    Authors:
        David Woodburn
    Version:
        1.2
    """

    # If the inputs (not T) are scalars, make them into 2D arrays.
    if np.ndim(F) == 0:
        F = np.array([[F]])
    if np.ndim(B) == 0:
        B = np.array([[B]])
    if np.ndim(Q) == 0:
        Q = np.array([[Q]])

    # Check the shapes of F, B, and Q.
    shape_F = F.shape
    shape_B = B.shape
    shape_Q = Q.shape
    Z1 = np.array([[0]])
    if (len(shape_F) != 2) or (len(shape_B) != 2) or (len(shape_Q) != 2):
        print("vanloan: Inputs must be 2D arrays or scalars!")
        return (Z1,Z1,Z1)
    n = shape_F[0]  # number of states
    k = shape_B[1]  # number of inputs

    # Get Phi.
    Phi = expm(F*T)

    # Get Bd.
    if (shape_F[0] != shape_B[0]):
        Bd = Z1
    else:
        Z = np.zeros((k, n+k))
        G = np.vstack(( np.hstack((F,B)), Z ))
        H = expm(G*T)
        Bd = H[0:n, n:(n+k)]

    # Get Qd.
    if ((shape_F[0] != shape_Q[0]) or (shape_F[1] != shape_Q[1])):
        Qd = Z1
    else:
        Z = np.zeros((n,n))
        L = np.vstack((
                np.hstack((-F, Q)),
                np.hstack(( Z, F.T))
            ))
        M = expm(L*T)
        Qd = Phi @ M[0:n, n:(2*n)]

    return (Phi, Bd, Qd)

def xcorr(x):
    """
    Calculate an estimate of the autocorrelation of x using a single
    realization.  The input x is expected to be a 1D array.  The return value is
    also a 1D array.  If the length of x is N, then the length of the return
    value will be 2N-1.  The return value is normalized such that the middle
    value will be 1.

    Dependencies:
        import numpy as np
    Authors:
        David Woodburn
    Version:
        1.0
    """

    N = len(x)
    C = np.zeros(2*N-1)
    for n in range(N):
        C[n] = np.sum(x[0:(n+1)]*x[(N-1-n):N])
    C[0:N] /= C[N-1]
    C[N:] = np.flip(C[0:(N-1)])
    return C

def avar(y, T):
    """
    This function calculates the Allan variance, va, over the averaging time
    array, tau, from the noise input signal, y.  Note that y is only the noise,
    not the signal.  If your y array has a dc value, you likely have not removed
    the signal.  The Allan variance (avar) is equal to

                             N - 2n
                             .----  .-                 -. 2
                       1      \     |                   |           tau
        va(tau) = ----------   }    | Y (tau) - Y (tau) |   :   n = --- ,
                  2 (N - 2n)  /     |  k+1       k      |            T
                             '----  '-                 -'
                             k = 1
    where
                         t + tau
                       .- k
                   1   |
        Y (tau) = ---  |  y(t) dt   :   T <= tau < N T/2 .
         k        tau  |
                      -' t
                          k

    Here, y is the noise input signal, N is the number of elements of y,
    Y_k(tau) is the average of chunk k of width tau of y, k is the chunk index,
    t_k is the time of the first element of chunk k, and T is the sampling
    period of y.

    First, we need to define an array, nn, of chunk widths (in terms of number
    of elements).  The minimum chunk width is one and the maximum chunk width is
    N/2 - 1.  We go only up to N/2 - 1 since we need to be able to access the k
    and k+1 chunks.  Since the Allan variance is commonly plotted with loglog
    scaling, we will use a logarithmic spread from 1 to N/2 - 1, flooring the
    result (since we need integer widths) and dropping non-unique values.  Then,
    the tau array is just this nn array times the sampling period, T.

    Second, the definition of Y_k(tau) requires an integral.  Rather than
    calculate an integral for each tau, we will calculate the cumulative sum of
    y:

        Y = cumsum(y) .

    We can rewrite Y_k(tau) and Y_(k+1)(tau) as

                   1  .-                 -.
        Y (tau) = --- |  Y(k+n)  -  Y(k)  |
         k         n  '-                 -'

                   1  .-                  -.
        Y (tau) = --- |  Y(k+2n) - Y(k+n)  | ,
         k+1       n  '-                  -'

    where instead of averaging over tau while scaling the cumulative sum by T,
    we cancel the sampling periods and just average over n, which is tau/T.
    This means that the difference


        delta = Y (tau) - Y (tau)
                 k+1       k

    is equal to

                 1  .-                           -.
        delta = --- |  Y(k+2n) - 2 Y(k+n) + Y(k)  |
                 n  '-                           -'

    Finally, our Allan variance for this particular chunk width n, would be

                             .---
                       1      \   .-       -. 2
        va(n T) = ----------   }  |  delta  |  .
                  2 (N - 2n)  /   '-       -'
                             '---

    This function can only handle 1D arrays.  Higher-dimensional y arrays will
    be flattened.

    References:
        IEEE Std 952-1997
        D. A. Howe, D. W. Allan, J. A. Barnes: "Properties of signal sources and
            measurement methods", pages 464-469, Frequency Control Symposium 35,
            1981.
        https://www.mathworks.com/help/nav/ug/
            inertial-sensor-noise-analysis-using-allan-variance.html
    Dependencies:
        import numpy as np
    Authors:
        Robert Leishman (original)
        David Woodburn (editor)
    Version:
        1.4
    """

    # Make sure the input is a 1D array.
    if (len(y.shape) > 1):
        y = y.flatten()

    # Define the averaging times.
    N = y.size # number of points in signal
    nn = np.unique(np.floor(np.logspace(0,np.log10(N/2-1),128))).astype(int)
    M = nn.size # number of chunks
    tau = nn*T # array of chunk sizes [s]

    # Get the cumulative sum of the input y.
    Y = np.cumsum(y)

    # Build the Allan variances array.
    va = np.zeros(M)
    for m, n in enumerate(nn): # m: chunk index.  n: chunk size
        k_max = N - 2*n # signal index of last chunk
        k = np.arange(k_max) # array of chunk starts
        deltas = (Y[k+2*n] - 2*Y[k+n] + Y[k])/n
        va[m] = np.sum(deltas**2)/(2*k_max)

    return va, tau

def armav(va, tau, log_scale=False):
    """
    This function solves for five common component noise variances to fit the
    given total Allan variance, va, as a function of averaging time, tau.  The
    component noise variances are
        - quantization
        - random walk
        - bias instability
        - rate random walk
        - rate ramp
    This algorithm uses the leastsq function to solve for the coefficients k in
    the equation

        y = H k,

    where k is the vector of component noise variances, y is the array of N
    Allan variances over time, and

            .-                                                  -.
            | 3/tau_1^2  1/tau_1  2 ln(2)/pi  tau_1/3  tau_1^2/2 |
        H = |    ...       ...       ...        ...       ...    | .
            | 3/tau_N^2  1/tau_N  2 ln(2)/pi  tau_N/3  tau_N^2/2 |
            '-                                                  -'

    Note that `ln' is the ISO standard notation for the logarithm base e
    (ISO/IEC 80000).

    The third input parameter `log_scale' is a boolean flag that controls
    whether fitting should be done in the log_scale base 10 scale or in the
    linear scale.

    References:
        IEEE Std 952-1997
        Jurado, Juan & Kabban, Christine & Raquet, John. (2019). A
            regression-based methodology to improve estimation of inertial
            sensor errors using Allan variance data. Navigation. 66.
            10.1002/navi.278.
    Dependencies:
        import numpy as np
        from scipy.optimize import leastsq
        import warnings
    Authors:
        David Woodburn
    Version:
        1.3
    """

    # Make sure all inputs are 1D arrays.
    if (len(va.shape) > 1):
        va = va.flatten()
    if (len(tau.shape) > 1):
        tau = tau.flatten()

    # linear regression matrix
    H = np.vstack([
        3/tau**2,
        1/tau,
        2*np.log(2)/np.pi + 0*tau,
        tau/3,
        tau**2/2]).T

    # fitting functions
    def flog(k, H, y):
        return np.log10(H @ np.absolute(k)) - y
    def flin(k, H, y):
        return (H @ np.absolute(k)) - y

    # least squares solver
    warnings.filterwarnings("ignore",
            message="Number of calls to function has reached maxfev")
    if log_scale:
        vl = np.log10(va)
        vk, _ = leastsq(flog, np.ones(5), args=(H, vl), maxfev=1000)
        vk = np.absolute(vk)
    else:
        vk, _ = leastsq(flin, np.ones(5), args=(H, va), maxfev=1000)
        vk = np.absolute(vk)

    return vk

def sysresample(w):
    """
    ii = sysresample(w)

    This function uses Systematic resampling based on particle weights.  Smaller
    weights have a lower probability of their corresponding indices being
    selected for the new array ii of sample indices.  This method of resampling
    in comparison with multinomial resampling, stratified resampling, and
    residual resampling was found to offer better resampling quality and lower
    computational complexity [1], [2].  An earlier reference to this method
    appears in [3].

    The cumulative sum, W, of the weights, w, is created.  Then a
    uniformly-spaced array, u, from 0 to (J-1)/J is created, where J is the
    number of weights.  To this array is added a single uniformly-distributed
    random number in the range [0, 1/J).  Within a loop over index j, the first
    value in the array W to exceed u[j] is noted.  The index of that value of W
    becomes the jth value in an array of indices ii.  In other words, the first
    weight to be responsible for pushing the cumulative sum of weights past a
    linearly growing value is likely a larger weight and should be reused.

        1 -|                            ..........----------````````````
           |                           /
           |                         /
           |                       /
           |                     /
           |         ........../
           |---``````
        0 -|---------+---------+---------+---------+---------+---------+
           0         1         2         3         4         5         6
                               ^
                               This weight contributed greatly to the cumulative
                               sum and is most likely to be selected at this
                               point.

    References:
        [1] Jeroen D. Hol, Thomas B. Schon, Fredrik Gustafsson, "On Resampling
            Algorithms For Particle Filters," presentated at the Nonlinear
            Statistical Signal Processing Workshop, 2006 IEEE. [Online].
            Available:
            http://users.isy.liu.se/rt/schon/Publications/HolSG2006.pdf
        [2] M. Sanjeev Arulampalam, Simon Maskell, Neil Gordon, and Tim Clapp,
            "A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian
            Bayesian Tracking," IEEE Transactions On Signal Processing, Vol. 50,
            No.  2, Feb. 2002. [Online]. Available:
            https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/optreadings/
        [3] Genshiro Kitagawa, "Monte Carlo Filter and Smoother for Non-Gaussian
            Nonlinear State Space Models," Journal of Computational and
            Graphical Statistics , Mar., 1996, Vol. 5, No. 1 (Mar., 1996), pp.
            1-25. [Online]. Available:
            https://www.jstor.org/stable/1390750?seq=1
    Dependencies:
        import numpy as np
    Authors:
        David Woodburn
    Version:
        1.0
    """

    J = len(w)
    W = np.cumsum(w)
    u = (np.arange(J) + np.random.uniform())/J
    i = 0
    ii = np.zeros(J)
    for j in range(J):
        while (W[i] < u[j]):
            i += 1
        ii[j] = i
    return ii

# ----------
# Simulation
# ----------

progress_len = 78
def progress(ratio):
    """
    Print a progress bar to the command window.  The width of the progress bar
    is set to 80 characters (including a beginning and an ending vertical bar).
    The input ratio should be a number between 0 and 1.  When, the ratio drops
    by more than one eightieth, the bar will reset its own counter.

    Dependencies:
        import numpy as np
    Authors:
        David Woodburn
    Version:
        1.2
    """

    global progress_len
    if (ratio*78 < progress_len - 1):
        progress_len = 0
        print("|", end = "", flush = True)
    else:
        if progress_len < 78:
            while np.ceil(ratio*78) > progress_len:
                progress_len += 1
                if progress_len < 78:
                    print("-", end = "", flush = True)
                else:
                    print("|", flush = True)

# -----
# Files
# -----

def bin_read_s4(file_name):
    """
    Read a binary file as an array of signed, 4-bit integers.  Parse each signed
    8-bit value into two, signed, 4-bit values assuming little-endianness.  This
    means that the less significant 4 bits in an 8-bit value are interpreted as
    coming sequentially before the most significant 4 bits in the 8-bit value.

    Dependencies:
        import numpy as np
    Authors:
        David Woodburn
    Version:
        1.0
    """

    x = np.fromfile(file_name, dtype=np.int8)
    x_even = ((np.bitwise_and(x, 0x0F) - 8) % 16) - 8
    x_odd = np.right_shift(x, 4)
    return np.array([x_even,x_odd], dtype=np.int8).flatten('F')

def bin_write_s4(x, file_name):
    """
    Write a binary file as an array of signed, 4-bit integers.  Combine pairs of
    two values as two, signed, 4-bit values into one 8-bit value, assuming
    little-endianness.  This means that the less significant 4 bits in an 8-bit
    value are interpreted as coming sequentially before the most significant 4
    bits in the 8-bit value.  The input x should already be scaled to a range of
    -7 to +7.  If it is not, the values will be clipped.

    Dependencies:
        import numpy as np
    Authors:
        David Woodburn
    Version:
        1.0
    """

    x[(x > 7)] = 7
    x[(x < -7)] = -7
    x = np.round(x).astype(np.int8)
    NX = int(np.ceil(len(x)/2))
    X = np.zeros(NX)
    X = np.left_shift(x[1::2], 4) + np.bitwise_and(x[0::2], 0x0F)
    X.tofile(file_name)
