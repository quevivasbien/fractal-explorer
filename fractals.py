#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:58:46 2020
Updated on Thu Jul 01 2021

@author: mckay
"""

import sys
import os
import numpy as np
import imageio
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
def gen_mandeltype(iterfunct,
                   xstart=-2, xstop=2, ystart=-2, ystop=2,
                   pixelwidth=3000, pixelheight=0,
                   maxiter=100, lim=10, break_lim=1e-10):
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
                # Note: the code will run very slowly if iterfunct is not
                #  defined with the @jit decorator
                last_z, z = z, iterfunct(z, c)
                i -= 1
                absz = abs(z)
                if absz > lim:
                    break
                if absz < break_lim or abs((abs(last_z) - absz) / absz) < break_lim:
                    i = 0
            # Save grid value as maxiter minus iterations to exceed lim,
            # grid values are normalized to be between -1 and 1
            # negative values mean that point is part of the mandelbrot set
            if i == 0:
                # scrunch to range (-1, 0)
                grid[row, col] = -((absz-2)/2)**4
            else:
                grid[row, col] = i / maxiter
    # Return values in interval (0, 1)
    return grid

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

def mandelbrot_scale(grid, maxiter):
    if COLORIZE:
        return construct_rgb_matrices(grid, RED, ORANGE, TAN, TEAL, GRAY)
    else:
        return (255 * grid).astype(np.uint8)


@jit
def basic_julia_iterfunct(z, c, n):
    return z**n + c


@jit
def gen_julia(c, n=2, iterfunct=basic_julia_iterfunct,
              xstart=-2, xstop=2, ystart=-2, ystop=2,
              pixelwidth=3000, maxiter=500, lim=5, invert=True):
    definition = (xstop - xstart) / pixelwidth
    x = np.arange(xstart, xstop, definition)
    y = np.arange(ystart, ystop, definition)
    grid = np.zeros((len(y), len(x)), dtype=np.uint16)
    # Iterate over grid and test for divergence at each pixel
    for row in range(len(y)):
        for col in range(len(x)):
            z = y[row]*1j + x[col]
            i = maxiter
            while i > 0 and abs(z) < lim:
                # Note: the code will run very slowly if iterfunct is not
                #  defined with the @jit decorator
                z = iterfunct(z, c, n)
                i -= 1
            # Save grid value as maxiter minus iterations to exceed lim
            grid[row][col] = i
    # Cube grid values to emphasize differences in divergence times
    grid = grid**np.log(maxiter+2)
    if invert:
        return (255*(1-grid/grid.max())).astype(np.uint8)
    else:
        return (255*(grid/grid.max())).astype(np.uint8)


@jit
def mandelbrot_iterfunct(z, c):
    return z**2 + c


def gen_mandelbrot(xstart=-2, xstop=1, ystart=-1.5, ystop=1.5,
                   pixelwidth=3000, pixelheight=0,
                   maxiter=100, lim=2, scale=True):
    grid = gen_mandeltype(iterfunct=mandelbrot_iterfunct,
                          xstart=xstart, xstop=xstop,
                          ystart=ystart, ystop=ystop,
                          pixelwidth=pixelwidth, pixelheight=pixelheight,
                          maxiter=maxiter, lim=lim)
    if scale:
        return mandelbrot_scale(grid, maxiter)
    else:
        return grid


def multicore_helper(args):
    xstart = args[0]
    xstop = args[1]
    ystart = args[2]
    ystop = args[3]
    pixelwidth = args[4]
    pixelheight = args[5]
    maxiter = args[6]
    lim = args[7]
    return gen_mandelbrot(xstart, xstop, ystart, ystop,
                          pixelwidth, pixelheight, maxiter, lim, scale=False)


def gen_mandelbrot_multicore(xstart=-2, xstop=1, ystart=-1.5, ystop=1.5,
                             pixelwidth=3000, pixelheight=0, maxiter=1000, lim=2,
                             n_processes=ncores):
    breaks = np.linspace(xstart, xstop, n_processes+1)
    pixelwidths = [pixelwidth // n_processes]*n_processes
    pixel_remainder = pixelwidth % n_processes
    if pixelheight == 0:
        pixelheight = int(pixelwidth * (ystop - ystart) / (xstop - xstart))
    for i in range(pixel_remainder):
        pixelwidths[i] += 1
    with Pool(n_processes) as p:
        subgrids = p.map(
                multicore_helper,
                [(breaks[i], breaks[i+1], ystart, ystop,
                  pixelwidths[i], pixelheight,
                  maxiter, lim) for i in range(n_processes)]
            )
    grid = np.concatenate(subgrids, axis=1)
    return mandelbrot_scale(grid, maxiter)


@cuda.jit
def mandel_iter(grid, x, y):
    i, j = cuda.grid(2)
    if i < grid.shape[0] and j < grid.shape[1]:
        c = y[i]*1j + x[j]
        z = 0
        k = 1000
        while k > 0:
            last_z, z = z, z**2 + c
            k -= 1
            absz = abs(z)
            if absz > 2:
                break
            if absz < 1e-10 or abs((abs(last_z) - absz) / absz) < 1e-10:
                k = 0
        if k == 0:
            # scrunch absz to range (-1, 0)
            grid[i, j] = -((absz-2)/2)**4
        else:
            grid[i, j] = k / 1000


def gen_mandelbrot_gpu(xstart=-2, xstop=2, ystart=-2, ystop=2,
                       pixelwidth=3000, pixelheight=0, scale=True,
                       threads_per_block=(32, 32)):
    if pixelheight == 0:
        pixelheight = int(pixelwidth * (ystop - ystart) / (xstop - xstart))
    x = np.linspace(xstart, xstop, pixelwidth)
    y = np.linspace(ystart, ystop, pixelheight)
    grid = np.empty((pixelheight, pixelwidth))
    # Iterate over grid and test for divergence at each pixel
    blocks_per_grid_x = np.ceil(pixelwidth / threads_per_block[1]).astype(int)
    blocks_per_grid_y = np.ceil(pixelheight / threads_per_block[0]).astype(int)
    mandel_iter[(blocks_per_grid_y, blocks_per_grid_x),
                   threads_per_block](grid, x, y)
    # Return values in interval (0, 1)
    return mandelbrot_scale(grid, 1000) if scale else grid


def save(grid, filename):
    imageio.imwrite(filename, grid)
        

def gen_mandeltype_mp4(expstart=2, expend=5, nframes=10,
                       xstart=-2, xstop=2, ystart=-2, ystop=2,
                       pixelwidth=500, maxiter=100, lim=2,
                       saveas='mandeltype.mp4', fps=4):
    frames = []
    for exponent in np.linspace(expstart, expend, nframes):
        @jit
        def iterfunct(z, c):
            return z**exponent + c
        frames.append(
            mandelbrot_scale(
            gen_mandeltype(iterfunct=iterfunct,
                           xstart=xstart, xstop=xstop,
                           ystart=ystart, ystop=ystop,
                           pixelwidth=pixelwidth, maxiter=maxiter, lim=lim
                           )), maxiter)
        print(f'Created frame {len(frames)} of {nframes}')
    imageio.mimwrite(uri=saveas, ims=frames, fps=fps)
    

def multiproc_julia(args):
    return gen_julia(args[0], args[1],
                     xstart=args[2], xstop=args[3],
                     ystart=args[4], ystop=args[5],
                     pixelwidth=args[6], maxiter=args[7], lim=args[8])
    

def gen_julia_rotation(radius, nframes, n=2,
                  xstart=-2, xstop=2, ystart=-2, ystop=2,
                  pixelwidth=500, maxiter=500, lim=5,
                  saveas='julia_rotation.mp4', fps=4,
                  n_processes=10):
    THETA = np.linspace(0, 2*np.pi, nframes, endpoint=False)
    frames = []
    while len(frames) < nframes:
        i = len(frames)
        nproc = n_processes if nframes - i >= n_processes else nframes - i
        with Pool(nproc) as p:
            frames += p.map(multiproc_julia,
                            [(radius*np.exp(THETA[i+s]*1j), n,
                             xstart, xstop,
                             ystart, ystop,
                             pixelwidth, maxiter, lim) for s in range(nproc)])
    width = min(f.shape[0] for f in frames)
    height = min(f.shape[1] for f in frames)
    imageio.mimwrite(uri=saveas,
                     ims=[f[:width, :height] for f in frames],
                     fps=fps)
    
    
def multiproc_mandelbrot(args):
    return gen_mandelbrot(xstart=args[0], xstop=args[1],
                          ystart=args[2], ystop=args[3],
                          pixelwidth=args[4], pixelheight=args[5],
                          maxiter=args[6])
    

def gen_mandelbrot_zoom(xrange1, yrange1, xrange0=(-2,1), yrange0=(-1.5, 1.5),
                        pixelwidth=512, maxiter=1000,
                        nframes=200, fps=30, saveas='mandelbrot_zoom.mp4',
                        n_processes=10, buffer_frames=30, zoom_factor=1e13):
    pixelheight = int(pixelwidth * (yrange0[1] - yrange0[0]) 
                      / (xrange0[1] - xrange0[0]))
    t = (np.logspace(0, -1, nframes, base=zoom_factor) - 1/zoom_factor) \
        / (1-1/zoom_factor)
    xstarts = t * xrange0[0] + (1-t) * xrange1[0]
    xends = t * xrange0[1] + (1-t) * xrange1[1]
    ystarts = t * yrange0[0] + (1-t) * yrange1[0]
    yends = t * yrange0[1] + (1-t) * yrange1[1]
    frames = []
    i = 0  # i is how many frames we've done so far
    while i < nframes:
        nproc = n_processes if nframes - i >= n_processes else nframes - i
        with Pool(nproc) as p:
            frames += p.map(multiproc_mandelbrot,
                            [(xstarts[i+s], xends[i+s],
                              ystarts[i+s], yends[i+s],
                              pixelwidth, pixelheight, maxiter) \
                                for s in range(nproc)])
        i += nproc
        print(f'Completed {i} of {nframes} frames')
    frames = [frames[0]]*buffer_frames + frames + [frames[-1]]*buffer_frames
    width = min(f.shape[0] for f in frames)
    height = min(f.shape[1] for f in frames)
    imageio.mimwrite(uri=saveas,
                     ims=[f[:width, :height] for f in frames],
                     fps=fps)

def gen_mandelbrot_zoom_gpu(xrange1, xrange2,
                            xrange0=(-2,1), yrange0=(-1.5, 1.5),
                            pixelwidth=512, nframes=200, fps=30,
                            save_loc='~/Documents/python/',
                            buffer_frames=30, zoom_factor=1e13,
                            keep_frames=False):
    save_loc = os.path.expanduser(save_loc)
    if not os.path.isdir(save_loc):
        os.mkdir(save_loc)
    pixelheight = int(pixelwidth * (yrange0[1] - yrange0[0]) 
                      / (xrange0[1] - xrange0[0]))
    t = (np.logspace(0, -1, nframes, base=zoom_factor) - 1/zoom_factor) \
        / (1-1/zoom_factor)
    xstarts = t * xrange0[0] + (1-t) * xrange1[0]
    xends = t * xrange0[1] + (1-t) * xrange1[1]
    ystarts = t * yrange0[0] + (1-t) * yrange1[0]
    yends = t * yrange0[1] + (1-t) * yrange1[1]
    frames = []
    for i in range(nframes):
        f = gen_mandelbrot_gpu(xstart=xstarts[i], xstop=xends[i],
                               ystart=ystarts[i], ystop=yends[i],
                               pixelwidth=pixelwidth, pixelheight=pixelheight)
        if keep_frames:
            if not os.path.isdir(os.path.join(save_loc, 'frames')):
                os.mkdir(os.path.join(save_loc, 'frames'))
            imageio.imwrite(uri=os.path.join(save_loc, 'frames', f'frame_{i}.png'),
                            im=f)
        frames.append(f)
        if i % 10 == 9:
            print(f'Completed {i+1} of {nframes} frames')
    imageio.mimwrite(uri=os.path.join(save_loc, 'zoom.mp4'), ims=frames, fps=fps)
        

if __name__ == '__main__' and len(sys.argv) > 1:
    args = sys.argv
    xrange1 = (float(args[1]), float(args[2]))
    yrange1 = (float(args[3]), float(args[4]))
    pixelwidth = int(args[5])
    nframes = int(args[6])
    save_loc = args[7]
    keep_frames = (args[8] == '1')
    gen_mandelbrot_zoom_gpu(xrange1, yrange1, (-64/27, 32/27), (-1, 1),
                            pixelwidth, nframes=nframes, save_loc=save_loc,
                            keep_frames=keep_frames)
