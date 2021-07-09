#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:56:05 2021

@author: mckay

Create Mandelbrot zooms or paths along Bezier curves
"""

import os
import numpy as np
import re
import imageio
from scipy.interpolate import interp1d
# from scipy.special import comb

from mandelbrot import MandelbrotMakerGPU

import matplotlib.pyplot as plt

# def bernstein_poly(k, n, t):
#     return comb(n, k) * t**k * (1-t)**(n-k)

# def bezier_curve(x, y, resolution=100):
#     n_points = len(x)
#     assert len(y) == n_points, 'x and y should have the same length'
#     t = np.linspace(0, 1, resolution)
#     poly_array = np.array([bernstein_poly(k, n_points-1, t) \
#                           for k in range(n_points)])
#     xout = np.dot(x, poly_array)
#     yout = np.dot(y, poly_array)
#     return xout, yout

def smooth_curve(x, y, resolution=100):
    """Uses cubic splines to create a smooth path in 2d"""
    assert len(x) == len(y), 'x and y should have the same length'
    t_ = np.linspace(0, 1, len(x))
    t = np.linspace(0, 1, resolution)
    xout = interp1d(t_, x, kind='cubic')(t)
    yout = interp1d(t_, y, kind='cubic')(t)
    return xout, yout


def gen_path_from_coords(pixelwidth=512, nframes=100,
                         coord_source='coordinates.txt',
                         save_loc='~/Documents/python/', keep_frames=False,
                         quality=10, fps=30, buffer_frames=30, colors=None):
    """Doesn't currently work properly"""
    # import and process the list of coordinates
    save_loc = os.path.expanduser(save_loc)
    if not os.path.isdir(save_loc):
        os.mkdir(save_loc)
    with open(coord_source, 'r') as fh:
        coords = fh.read().strip().split('\n')
    searches = [re.search(r'^re (-?[\d\.]+)_(-?[\d\.]+) im (-?[\d\.]+)_(-?[\d\.]+)$', x) \
                for x in coords]
    xstarts = [float(s.group(1)) for s in searches]
    xstops = [float(s.group(2)) for s in searches]
    ystarts = [float(s.group(3)) for s in searches]
    ystops = [float(s.group(4)) for s in searches]
    # figure out path
    width = xstops[0] - xstarts[0]
    height = ystops[0] - ystarts[0]
    xstart_path, ystart_path = smooth_curve(xstarts, ystarts, nframes)
    xstop_path, ystop_path = xstart_path + width, ystart_path + height
    plt.plot(xstart_path, ystart_path)
    plt.scatter(xstarts, ystarts)
    plt.plot(xstop_path, ystop_path)
    plt.scatter(xstops, ystops)
    plt.show()
    return
    pixelheight = int(pixelwidth * height / width)
    if colors is None:
        mm = MandelbrotMakerGPU()
    else:
        mm = MandelbrotMakerGPU(c0=colors[0], c1=colors[1], c2=colors[2],
                                c3=colors[3], c4=colors[4])
    frames = []
    for i in range(0, nframes):
        if i == 0:
            f= mm.gen_mandelbrot(xstart_path[i], xstop_path[i],
                                 ystart_path[i], ystop_path[i],
                                 pixelwidth=pixelwidth, pixelheight=pixelheight)
        else:
            # recycle the last frame
            # what a mess;
            # I should probably put this in its own function and comment heavily
            # TODO: deal with zero or negative matrix sizes
            last = frames[i-1]
            last_width = xstop_path[i-1] - xstart_path[i-1]
            last_height = ystop_path[i-1] - ystart_path[i-1]
            if xstart_path[i] > xstart_path[i-1]:  # move to right
                xmove = 0
                lost_width = xstart_path[i] - xstart_path[i-1]
                lost_pixelwidth = int(last.shape[1] * lost_width / last_width)
                keep = last[:, lost_pixelwidth:]
                xstart, xstop = xstop_path[i-1], xstop_path[i]
            else:  # move left
                xmove = 1
                shared_width = xstop_path[i] - xstart_path[i-1]
                shared_pixelwidth = int(last.shape[1] * shared_width / last_width)
                keep = last[:, :shared_pixelwidth]
                xstart, xstop = xstart_path[i], xstart_path[i-1]
            if ystart_path[i] > ystart_path[i-1]:  # move down
                ymove = 0
                lost_height = ystart_path[i] - ystart_path[i-1]
                lost_pixelheight = int(last.shape[0] * lost_height / last_height)
                keep = keep[lost_pixelheight:, :]
                ystart, ystop = ystop_path[i-1], ystop_path[i]
            else:  # move up
                ymove = 1
                shared_height = ystop_path[i] - ystart_path[i-1]
                shared_pixelheight = int(last.shape[0] * shared_height / last_height)
                keep = keep[:shared_pixelheight, :]
                ystart, ystop = ystart_path[i], ystart_path[i-1]
            # assemble new vertical component
            pxwidth = last.shape[1] - keep.shape[1]
            if pxwidth > 0:
                vert = mm.gen_mandelbrot(xstart, xstop,
                                         ystart_path[i], ystop_path[i],
                                         pixelwidth=pxwidth,
                                         pixelheight=last.shape[0])
            else:
                vert = None
            # assemble new horizontal component
            pxheight = last.shape[0] - keep.shape[0]
            if pxheight > 0:
                if xmove == 0:
                    horiz = mm.gen_mandelbrot(xstart_path[i], xstart,
                                              ystart, ystop,
                                              pixelwidth=keep.shape[1],
                                              pixelheight=pxheight)
                else:
                    horiz = mm.gen_mandelbrot(xstop, xstop_path[i],
                                              ystart, ystop,
                                              pixelwidth=keep.shape[1],
                                              pixelheight=pxheight)
            else:
                horiz = None
            # now put everything together
            if vert is not None and horiz is not None:
                if xmove == 0 and ymove == 1:
                    f = np.concatenate((np.concatenate((horiz, keep), 0), vert), 1)
                elif xmove == 0 and ymove == 0:
                    f = np.concatenate((np.concatenate((keep, horiz), 0), vert), 1)
                elif xmove == 1 and ymove == 1:
                    f = np.concatenate((vert, np.concatenate((horiz, keep), 0)), 1)
                else:
                    f = np.concatenate((vert, np.concatenate((keep, horiz), 0)), 1)
            # Deal with cases where we don't need one of the slices
            elif vert is None and horiz is None:
                f = keep
            elif vert is None:
                if ymove == 0:
                    f = np.concatenate((keep, horiz), 0)
                else:
                    f = np.concatenate((horiz, keep), 0)
            else:
                if xmove == 0:
                    f = np.concatenate((keep, vert), 1)
                else:
                    f = np.concatenate((vert, keep), 1)
        # save frame
        if keep_frames:
            if not os.path.isdir(os.path.join(save_loc, 'frames')):
                os.mkdir(os.path.join(save_loc, 'frames'))
            imageio.imwrite(uri=os.path.join(save_loc, 'frames', f'frame_{i}.png'),
                            im=f)
        frames.append(f)
        if i % 10 == 9:
            print(f'Completed {i+1} of {nframes} frames')
    frames = [frames[0]]*buffer_frames + frames + [frames[-1]]*buffer_frames
    imageio.mimwrite(os.path.join(save_loc, 'bezier_path.mp4'),
                     frames, fps=fps, quality=quality)
    