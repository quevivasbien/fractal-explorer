This package includes a python GUI `fractal_explorer.py` that allows the user to explore the Mandelbrot set. It also includes some additional functions in `fractals.py` that can be used to generate other fractal types related to the Mandelbrot set (e.g. Julia sets). The code supports multiprocessing and GPU processing with CUDA. `mandelbrot.py` includes a more versatile implementation of the Mandelbrot set functions in `fractals.py`. `zooms.py` includes functions for creating videos based on the Mandelbrot set.

To use the fractal explorer GUI, launch `fractal_explorer.py` with arguments to determine the horizontal resolution and whether to use the GPU instead of the CPU:

```python
python fractal_explorer.py [pixelwidth] [use_gpu]
```

`pixelwidth` should be an integer, and `use_gpu` should be either 0 (use CPU) or 1 (use GPU).

---

Requirements: `numpy`, `numba`, `cudatoolkit`, `PySimpleGUI`
