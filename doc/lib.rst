Processing Library
==================

Functions for processing laser image data.

Calculations
------------

This module mainly contains mathmatical functions used in other modules.

.. automodule:: pew.lib.calc
    :members:


Colocalisation
--------------

Colocalisation can be used to quanitify the positional relationship between
elements. Various algorithms exist and a few of them are implemented in this module.

.. automodule:: pew.lib.colocal
    :members:


Convolution
-----------

The are many applications for convolution and deconvolution of images such as
blurring an image or removing wash-out tailing. This module contains functions
for 1-dimensional convolution as well as functions for creating distributions.

.. automodule:: pew.lib.convolve
    :members:


Filtering
---------

.. automodule:: pew.lib.filters
    :members:


Peakfinding
-----------

This module is under devolopment.

Thresholding
------------

Currently this module only contains a reimplementation of Otsu's method.

.. automodule:: pew.lib.threshold
    :members:
