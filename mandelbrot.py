#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 19:29:04 2021

@author: mckay
"""

import numpy as np
from numba import jit, cuda
from multiprocessing import Pool, cpu_count

COLORIZE = True

TEAL = (84, 147, 146)
GRAY = (178, 182, 183)
TAN = (195, 155, 114)
ORANGE = (212, 135, 40)
RED = (190, 67, 66)

ncores = cpu_count()


@jit
def gen_mandelbrot_(xstart=-2, xstop=1, ystart=-1.5, ystop=1.5,
                    pixelwidth=3000, pixelheight=0, maxiter=1000, 
                    break_lim=1e-10, scale=True):
    """Generates Mandelbrot set array using a single CPU core"""
    if pixelheight == 0:
        pixelheight = int(pixelwidth * (ystop - ystart) / (xstop - xstart))
    x = np.linspace(xstart, xstop, pixelwidth)
    y = np.linspace(ystart, ystop, pixelheight)
    grid = np.zeros((pixelheight, pixelwidth))
    # Iterate over grid and test for divergence at each pixel
    for row in range(len(y)):
        for col in range(len(x)):
            c = y[row]*1j + x[col]
            z = 0
            i = maxiter
            while i > 0:
                last_z, z = z, z**2 + c
                i -= 1
                absz = abs(z)
                if absz > 2:
                    break
                if absz < break_lim \
                    or abs((abs(last_z) - absz) / absz) < break_lim:
                    i = 0
            # Save grid value as maxiter minus iterations to exceed lim,
            # grid values are normalized to be between -1 and 1
            # negative values mean that point is part of the mandelbrot set
            if i == 0:
                # scrunch to range (-1, 0)
                grid[row, col] = -((absz-2)/2)**4
            else:
                grid[row, col] = i / maxiter
    # Return values in interval (-1, 1)
    return grid


@cuda.jit
def cuda_mandeliter(grid, x, y, maxiter=1000, break_lim=1e-10):
    i, j = cuda.grid(2)
    if i < grid.shape[0] and j < grid.shape[1]:
        c = y[i]*1j + x[j]
        z = 0
        k = maxiter
        while k > 0:
            last_z, z = z, z**2 + c
            k -= 1
            absz = abs(z)
            if absz > 2:
                break
            if absz < break_lim \
                or abs((abs(last_z) - absz) / absz) < break_lim:
                k = 0
        if k == 0:
            # scrunch absz to range (-1, 0)
            grid[i, j] = -((absz-2)/2)**4
        else:
            grid[i, j] = k / maxiter


@jit
def bw_to_quintagradient(x, c0, c1, c2, c3, c4):
    """Takes a value x between -1 and 1 and outputs (r, g, b) colors"""
    if x < 0:
        y = -x
        return (y * c0[0] + (1-y) * c1[0],
                y * c0[1] + (1-y) * c1[1],
                y * c0[2] + (1-y) * c1[2])
    elif x < 0.25:
        y = x * 4
        return (y * c2[0] + (1-y) * c1[0],
                y * c2[1] + (1-y) * c1[1],
                y * c2[2] + (1-y) * c1[2])
    elif x < 0.5:
        y = (x - 0.25) * 4
        return (y * c3[0] + (1-y) * c2[0],
                y * c3[1] + (1-y) * c2[1],
                y * c3[2] + (1-y) * c2[2])
    else:
        y = x * 2 - 1
        return (y * c4[0] + (1-y) * c3[0],
                y * c4[1] + (1-y) * c3[1],
                y * c4[2] + (1-y) * c3[2])

@jit
def construct_rgb_matrices(grid, c0, c1, c2, c3, c4):
    red = np.empty(grid.shape)
    green = np.empty(grid.shape)
    blue = np.empty(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            red[i, j], green[i, j], blue[i, j] \
                = bw_to_quintagradient(grid[i, j], c0, c1, c2, c3, c4)
    return np.stack((red.astype(np.uint8),
                     green.astype(np.uint8),
                     blue.astype(np.uint8)), axis=-1)



class MandelbrotMaker:
    
    def __init__(self, maxiter=1000, break_lim=1e-10, colorize=True,
                 c0=RED, c1=ORANGE, c2=TAN, c3=TEAL, c4=GRAY):
        self.maxiter = maxiter
        self.break_lim = break_lim
        self.colorize = colorize
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
    
    def scale(self, grid):
        if self.colorize:
            return construct_rgb_matrices(grid, self.c0, self.c1,
                                          self.c2, self.c3, self.c4)
        else:
            return (255 * grid).astype(np.uint8)
        
    
    def multicore_helper(self, args):
        xstart, xstop, ystart, ystop, pixelwidth, pixelheight = args
        return gen_mandelbrot_(xstart=xstart, xstop=xstop,
                               ystart=ystart, ystop=ystop,
                               pixelwidth=pixelwidth, pixelheight=pixelheight,
                               maxiter=self.maxiter, break_lim=self.break_lim,
                               scale=False)
    
    def gen_mandelbrot(self, xstart=-2, xstop=1, ystart=-1.5, ystop=1.5,
                       pixelwidth=3000, pixelheight=0, n_processes=ncores,
                       scale=True):
        breaks = np.linspace(xstart, xstop, n_processes+1)
        pixelwidths = [pixelwidth // n_processes]*n_processes
        pixel_remainder = pixelwidth % n_processes
        if pixelheight == 0:
            pixelheight = int(pixelwidth * (ystop - ystart) / (xstop - xstart))
        for i in range(pixel_remainder):
            pixelwidths[i] += 1
        with Pool(n_processes) as p:
            subgrids = p.map(
                    self.multicore_helper,
                    [(breaks[i], breaks[i+1], ystart, ystop,
                      pixelwidths[i], pixelheight) for i in range(n_processes)]
                )
        grid = np.concatenate(subgrids, axis=1)
        return self.scale(grid) if scale else grid



class MandelbrotMakerGPU(MandelbrotMaker):
    
    def multicore_helper(self, args):
        raise NotImplementedError
    
    def gen_mandelbrot(self, xstart=-2, xstop=1, ystart=-1.5, ystop=1.5,
                       pixelwidth=3000, pixelheight=0,
                       threads_per_block=(8, 8), scale=True):
        if pixelheight == 0:
            pixelheight = int(pixelwidth * (ystop - ystart) / (xstop - xstart))
        x = np.linspace(xstart, xstop, pixelwidth)
        y = np.linspace(ystart, ystop, pixelheight)
        grid = np.empty((pixelheight, pixelwidth))
        blocks_per_grid_x = np.ceil(
                pixelwidth / threads_per_block[1]
            ).astype(int)
        blocks_per_grid_y = np.ceil(
                pixelheight / threads_per_block[0]
            ).astype(int)
        cuda_mandeliter[(blocks_per_grid_y, blocks_per_grid_x),
                        threads_per_block](grid, x, y,
                                           self.maxiter, self.break_lim)
        return self.scale(grid) if scale else grid
    
    