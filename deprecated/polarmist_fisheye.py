from copy import deepcopy
import os
import pickle
import time
import cv2
from matplotlib import pyplot as plt
import matplotlib



matplotlib.use("TkAgg")
import numpy as np
from numpy import sin, cos
from numba import njit
import scipy.interpolate as interp

X_C = 639.5
Y_C = 359.5

class FishCalib:
    def __init__(self, radius, aperture, poly_r_denorm2r_ideal, reverse_degree=25, sq_size=500):
        self.radius = radius
        self.aperture = aperture
        self.poly_r_denorm2r_ideal = poly_r_denorm2r_ideal
        x = np.linspace(0, 1)
        y = np.polyval(poly_r_denorm2r_ideal, x)
        self.poly_r_ideal2r_denorm = np.polyfit(y, x, reverse_degree)

        self.sq_size = sq_size
        self.image_shape = (720, 1280)
        # self.cubemaps = self.gen_cubemaps()
        # self.eqmaps = self.gen_eqmaps()
        # r = self.poly_r_ideal2r_denorm
        # self.poly_r_ideal2r_denorm = self.poly_r_denorm2r_ideal
        # self.poly_r_denorm2r_ideal = r

    def r_ideal2r_denorm(self, r):
        return np.polyval(self.poly_r_ideal2r_denorm, r)
  
    def r_denorm2r_ideal(self, r):
        return np.polyval(self.poly_r_denorm2r_ideal, r)
  
    def fishdenorm2fishideal(self, xd, yd):
        x = (xd - X_C) / self.radius
        y = (yd - Y_C) / self.radius
        theta = np.arctan2(y, x)
        r = (x**2 + y**2)**0.5
   

        # assert r.all() <= 1
        rnew = self.r_denorm2r_ideal(r)
        xnew = np.cos(theta) * rnew
        ynew = np.sin(theta) * rnew
        return xnew, ynew

    def fishideal2fishdenorm(self, xi, yi):
        theta = np.arctan2(yi, xi)
        r = (xi**2 + yi**2)**0.5
        r_denorm = self.r_ideal2r_denorm(r)
        # r_denorm = r
        # Convert back to Cartesian coordinates
        assert r.all() <= 1
        x = np.cos(theta) * r_denorm
        y = np.sin(theta) * r_denorm

        # Scale and translate back to pixel coordinates
        xd = x * self.radius + X_C
        yd = y * self.radius + Y_C
        return np.round(xd), np.round(yd)

    @staticmethod
    def fishideal2xyz(xn, yn, aperture, R=None):
        phi_f = (xn**2 + yn**2)**0.5 * aperture / 2
        theta_f = np.arctan2(yn, xn)

        x = np.sin(phi_f)*np.cos(theta_f)
        y = np.cos(phi_f)
        z = np.sin(theta_f)*np.sin(phi_f)
        if R is None or R.shape != (3,3):
            return x, y, z
        else:
            if not isinstance(x, float):
                xyz = np.vstack((x, y, z))
                xyzR = R.dot(xyz)
                return xyzR[0,:], xyzR[1,:], xyzR[2,:]
            else:
                xyzR = R.dot(np.array((x, y, z)))
                return xyzR[0], xyzR[1], xyzR[2]

    def xyz2fishideal(self, x, y, z):
        r = 2 * np.arctan2((x**2 + z**2)**0.5, y) / self.aperture
        theta = np.arctan2(z, x)
        xi = r * cos(theta)
        yi = r * sin(theta)
        return xi, yi

    def xyz2fishdenorm(self, x, y, z):
        xi, yi = self.xyz2fishideal(x, y, z)
        xd, yd = self.fishideal2fishdenorm(xi, yi)
        return xd, yd

    def fishdenorm2xyz(self, x, y, R=None):
        xi, yi = self.fishdenorm2fishideal(x, y)
        return self.fishideal2xyz(xi, yi, self.aperture, R=R)

    def fishdenorm2thetaphi(self, x, y, R=None):
        return self.xyz2thetaphi(*self.fishdenorm2xyz(x, y, R=R))

    @staticmethod
    def xyz2thetaphi(x, y, z):
        theta = np.arctan2(y, x) % (2*np.pi)
        phi = np.arccos(z)
        return theta, phi

    @staticmethod
    def thetaphi2xyz(theta, phi, R=None):
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        if R is not None:
            xyz = np.hstack((x.reshape((-1,1)), y.reshape((-1,1)), z.reshape((-1,1)))).T
            xyzR = R.dot(xyz)
            x, y, z = xyzR[0,:], xyzR[1,:], xyzR[2,:]
        return x, y, z

    @staticmethod
    def cube2xyz(face, t, s):
        assert face in [1, 2, 3, 4]
        if face == 1:
            x = (1 + t) / 2**0.5
            y = (1 - t) / 2**0.5
            z = s
        elif face == 2:
            x = (-1 + t) / 2**0.5
            y = (1 + t) / 2**0.5
            z = s
        elif face == 3:
            x = (-1 - t) / 2**0.5
            y = (-1 + t) / 2**0.5
            z = s
        elif face == 4:
            x = (1 - t) / 2**0.5
            y = (-1 - t) / 2**0.5
            z = s

        r = (x**2 + y**2 + z**2)**0.5
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)

        x = sin(phi)*cos(theta)
        y = sin(phi)*sin(theta)
        z = cos(phi)
        # rnew = (x**2 + y**2 + z**2)**0.5
        # assert abs(rnew - 1) < 1e-5, f"t: {t}, s: {s}, r: {r}"
        return x, y, z
    
    def cube_pixel2xyz(self, xpixel, ypixel):
        face = self._xpixel2face(xpixel)
        t = self._xpixel2t(face, xpixel)
        s = ypixel / (self.sq_size - 1) * 2 - 1
        return self.cube2xyz(face, t, s)

    

    def _xpixel2face(self, xpixel):
        if xpixel >= self.sq_size:
            face = 1
        else:
            face = 2
        return face

    def _xpixel2t(self, face, xpixel):
        t_denorm = xpixel - (2 - face) * self.sq_size
        return t_denorm / self.sq_size * 2 - 1
    
    def gen_cubemaps(self):
        cache_path = f"cubemap_{self.sq_size}.pickle"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
            
        sq_size = self.sq_size
        out_shape = (sq_size, 2*sq_size)
        mapx = np.zeros(out_shape, dtype="float32")
        mapy = np.zeros(out_shape, dtype="float32")
        for y_target in range(sq_size):
            for x_target in range(2*sq_size):
                face = self._xpixel2face(x_target)
                t = self._xpixel2t(face, x_target)
                s = y_target / (sq_size - 1) * 2 - 1
                assert -1 <= s <= 1
                x, y, z = self.cube2xyz(face, t, s) # points on sphere
                xyzR = np.array((x, y, z))
                x_source, y_source = self.xyz2fishdenorm(xyzR[0], xyzR[1], xyzR[2])
                mapx[y_target, x_target] = x_source
                mapy[y_target, x_target] = y_source
        with open(cache_path, "wb") as f:
            pickle.dump((mapx, mapy), f)
        return mapx, mapy
    
    def image2cubemap(self, image):
        return cv2.remap(image, *self.cubemaps, cv2.INTER_LINEAR)
    
    def mask2cubemap(self, mask):
        mask_cube = cv2.remap(mask, *self.cubemaps, cv2.INTER_LINEAR)
        mask_cube = (mask_cube > 100).astype(np.uint8)*255
        return mask_cube
    
    def image2eqmap(self, image):
        return cv2.remap(image, *self.eqmaps, cv2.INTER_LINEAR)
    
    def gen_eqmaps(self, R=None):
        cache_path = f"eqmap_{self.sq_size}.pickle"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        sq_size = self.sq_size
        shape = (sq_size, 2*sq_size)
        mapx = np.zeros(shape, dtype="float32")
        mapy = np.zeros(shape, dtype="float32")

        for y_target in range(sq_size):
            for x_target in range(2*sq_size):
                phi = y_target / (sq_size - 1) * np.pi / 2 + np.pi / 4
                theta = x_target / (2*sq_size - 1) * np.pi
                # phi = np.pi / 4 - phi
                x, y, z = self.thetaphi2xyz(theta, phi)
                xp, yp = self.xyz2fishdenorm(x, y, z)
                if not 0 <= theta <= np.pi:
                    continue
                if not np.pi/4 <= phi <= 3*np.pi/4:
                    continue
                y_target = sq_size - 1 - y_target
                x_target = 2*sq_size - 1 - x_target
                mapx[y_target, x_target] = xp
                mapy[y_target, x_target] = yp
        with open(cache_path, "wb") as f:
            pickle.dump((mapx, mapy), f)
        return mapx, mapy
 

# fish_calib = FishCalib(662, 200 * np.pi / 180, np.array([1.76838231e+00, -5.98990597e+00, 8.46504924e+00, -6.30470184e+00, 2.64910542e+00, -5.78675524e-01, 7.32021836e-02, 9.17391259e-01, 4.24473128e-05]))
fish_calib = FishCalib(662, 200 * np.pi / 180, np.array([0.016248739491802746, 0.018858310252328498, 0.01971739099626771, 0.007914765999675276, 0.934584334780868, 0.0]))
# fish_calib = FishCalib(662, 200 * np.pi / 180, np.array([0.016248739491802746, 0.018858310252328498, 0.01971739099626771, 0.007914765999675276, 0.934584334780868, 0.0]))
# fish_calib = FishCalib(662, 200 * np.pi / 180, np.array([0.29606927, -0.40027161, 0.01971739099626771, 0.21585077, 0.88411871, 0.0]))
# fish_calib = FishCalib(662, 200 * np.pi / 180, np.array([0.29606927, -0.40027161, 0.21585077, 0.88411871, 0.0]))
# fish_calib = FishCalib(662, 200 * np.pi / 180, np.array([0.002859019167036881, 0.08478604768556876, -0.09197053319834321,  0.9921667132593568, 0.0]))