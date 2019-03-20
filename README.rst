|CI|_ |Python|_ |License|_

.. |CI| image:: https://circleci.com/gh/tupui/otsensitivity.svg?style=svg
.. _CI: https://circleci.com/gh/tupui/otsensitivity

.. |Python| image:: https://img.shields.io/badge/python-2.7,_3.7-blue.svg
.. _Python: https://python.org

.. |License| image:: https://img.shields.io/badge/license-LGPL-blue.svg
.. _License: https://opensource.org/licenses/LGPL

otSensitivity
=============

What is it?
-----------

This project implements Sensitivity Analysis methods.
It is based on `OpenTURNS <http://www.openturns.org>`_.

Example: 

.. code-block:: python

    s, st = sobol_saltelli(model, 1000, 3, [[-np.pi, -np.pi, -np.pi],
                                            [np.pi, np.pi, np.pi]])
    
    
    

The output is the following figure: 

.. image::  doc/images/npfda-elnino-OutlierTrajectoryPlot.png


How to install?
---------------

Requirements
............

The dependencies are: 

- Python >= 2.7 or >= 3.3
- `numpy <http://www.numpy.org>`_ >= 0.10
- `scipy <http://scipy.org>`_ >= 0.15
- `OpenTURNS <http://www.openturns.org>`_ >= 1.12
- `matplotlib <https://matplotlib.org>`_ >= 1.5.3


Installation
............

Using the latest python version is prefered! Then to install::

    git clone git@github.com:.../otsensitivity.git
    cd otsensitivity
    python setup.py install
