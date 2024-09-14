# __init__.py for modeling_gui package

# Import key modules and classes
from .models import ModelManager
from .visualization import (
    plot_data,
    plot_regression,
    plot_confusion_matrix,
    plot_tree_diagram,
    plot_curve_fit
)
from .dialogs import RandomForestDialog, GradientBoostDialog, KMeansDialog

# Version of the modeling_gui package
__version__ = '1.0.0'

# Specify all imports when using "from modeling_gui import *"
__all__ = [
    'ModelManager',
    'plot_data',
    'plot_regression',
    'plot_confusion_matrix',
    'plot_tree_diagram',
    'plot_curve_fit',
    'RandomForestDialog',
    'GradientBoostDialog',
    'KMeansDialog',
]

