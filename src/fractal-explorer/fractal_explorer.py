#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:13:18 2021

@author: mckay
"""

import sys
import base64
import PySimpleGUI as sg
import numpy as np
from PIL import Image
from io import BytesIO

from fractals import gen_mandelbrot_multicore, gen_mandelbrot_gpu

### Engine for moving around in the fractal space

class ImageEngine:
    
    def __init__(self, pixelwidth, use_gpu=False,
                 xstart=-2, xstop=1, ystart=-1.5, ystop=1.5):
        self.pixelwidth = pixelwidth
        self.use_gpu = use_gpu
        self.xstart = xstart
        self.xstop = xstop
        self.ystart = ystart
        self.ystop = ystop
    
    def gen_grid(self):
        if self.use_gpu:
            return gen_mandelbrot_gpu(self.xstart, self.xstop,
                                      self.ystart, self.ystop,
                                      pixelwidth=self.pixelwidth)
        else:
            return gen_mandelbrot_multicore(self.xstart, self.xstop,
                                            self.ystart, self.ystop,
                                            pixelwidth=self.pixelwidth,
                                            maxiter=1000)
    
    def get_image_str(self, regen=True):
        if regen:
            self.grid = self.gen_grid()
        buffered = BytesIO()
        Image.fromarray(self.grid).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str
    
    def move_up(self):
        delta = self.ystop - self.ystart
        new_ystart = self.ystart - 0.25 * delta
        helper = ImageEngine(self.pixelwidth, self.use_gpu,
                             self.xstart, self.xstop,
                             new_ystart, self.ystart)
        helper_grid = helper.gen_grid()
        stop_row = -helper_grid.shape[0]
        self.grid = np.concatenate((helper_grid, self.grid[:stop_row, :]), axis=0)
        self.ystart = new_ystart
        self.ystop -= 0.25 * delta
        return self.get_image_str(regen=False)
    
    def move_down(self):
        delta = self.ystop - self.ystart
        new_ystop = self.ystop + 0.25 * delta
        helper = ImageEngine(self.pixelwidth, self.use_gpu,
                             self.xstart, self.xstop,
                             self.ystop, new_ystop)
        helper_grid = helper.gen_grid()
        start_row = helper_grid.shape[0]
        self.grid = np.concatenate((self.grid[start_row:, :], helper_grid), axis=0)
        self.ystop = new_ystop
        self.ystart += 0.25 * delta
        return self.get_image_str(regen=False)
    
    def move_left(self):
        delta = self.xstop - self.xstart
        new_xstart = self.xstart - 0.25 * delta
        helper = ImageEngine(self.pixelwidth // 4, self.use_gpu,
                             new_xstart, self.xstart,
                             self.ystart, self.ystop)
        helper_grid = helper.gen_grid()
        stop_col = -helper_grid.shape[1]
        self.grid = np.concatenate((helper_grid, self.grid[:, :stop_col]), axis=1)
        self.xstart = new_xstart
        self.xstop -= 0.25 * delta
        return self.get_image_str(regen=False)
    
    def move_right(self):
        delta = self.xstop - self.xstart
        new_xstop = self.xstop + 0.25 * delta
        helper = ImageEngine(self.pixelwidth // 4, self.use_gpu,
                             self.xstop, new_xstop,
                             self.ystart, self.ystop)
        helper_grid = helper.gen_grid()
        start_col = helper_grid.shape[1]
        self.grid = np.concatenate((self.grid[:, start_col:], helper_grid), axis=1)
        self.xstop = new_xstop
        self.xstart += 0.25 * delta
        return self.get_image_str(regen=False)
    
    def zoom_in(self):
        if (self.xstop - self.xstart) < 1e-13:
            # Reached zoom limit due to floating point precision
            return self.get_image_str(regen=False)
        xmedian = (self.xstop + self.xstart) / 2
        ymedian = (self.ystop + self.ystart) / 2
        self.xstop -= (0.75 * (self.xstop - xmedian))
        self.xstart += (0.75 * (xmedian - self.xstart))
        self.ystop -= (0.75 * (self.ystop - ymedian))
        self.ystart += (0.75 * (ymedian - self.ystart))
        return self.get_image_str()
    
    def zoom_out(self):
        xmedian = (self.xstop + self.xstart) / 2
        ymedian = (self.ystop + self.ystart) / 2
        self.xstop += (3 * (self.xstop - xmedian))
        self.xstart -= (3 * (xmedian - self.xstart))
        self.ystop += (3 * (self.ystop - ymedian))
        self.ystart -= (3 * (ymedian - self.ystart))
        return self.get_image_str()
    
    def get_coords(self):
        return f're {self.xstart}_{self.xstop} ' \
            + f'im {self.ystart}_{self.ystop}'
            

### main function controls user interface

def main(pixelwidth=1000, use_gpu=False):
    imageEngine = ImageEngine(pixelwidth=pixelwidth, use_gpu=use_gpu)
    image = sg.Image(data=imageEngine.get_image_str())
    layout = [
            [image],
            [sg.Button('Up')],
            [sg.Button('Left'), sg.Button('Right')],
            [sg.Button('Down')],
            [sg.Button('-'), sg.Button('+')],
            [sg.Button('Save')]
        ]
    window = sg.Window('Fractal Explorer', layout, element_justification='c')
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Up':
            image.update(data=imageEngine.move_up())
        elif event == 'Down':
            image.update(data=imageEngine.move_down())
        elif event == 'Left':
            image.update(data=imageEngine.move_left())
        elif event == 'Right':
            image.update(data=imageEngine.move_right())
        elif event == '+':
            image.update(data=imageEngine.zoom_in())
        elif event == '-':
            image.update(data=imageEngine.zoom_out())
        elif event == 'Save':
            Image.fromarray(imageEngine.grid).save(
                    imageEngine.get_coords() + '.png'
                )
    window.close()


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        try:
            assert args[2] in ('0', '1')
            pixelwidth = int(args[1])
            use_gpu = bool(int(args[2]))
        except:
            print('Syntax is "python fractal_explorer.py [pixelwidth] [use_gpu (0/1)]"')
            raise
        main(pixelwidth, use_gpu)
    else:
        main()
        