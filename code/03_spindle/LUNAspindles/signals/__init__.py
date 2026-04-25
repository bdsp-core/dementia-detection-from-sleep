# -*- coding: utf-8 -*-
"""
LUNAspindles.signals
-------
A python toolbox for processing EEG signals using LUNA (http://zzz.bwh.harvard.edu/luna/)
Prepares files for LUNA and generate sleep spindle features (micro-features)
(c) 2019 by Noor Adra (CDAC Westover Lab, MGH)
license?
"""

# compat
from __future__ import absolute_import, division, print_function

# allow lazy loading
from LUNAspindles import tools
from LUNAspindles.run import main, check_annots
from . import ecg_correction, extract_signals, read_signals
