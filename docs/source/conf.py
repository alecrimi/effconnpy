
Copy
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust the path to your project root

project = 'effconnpy'
author = 'Alessandro Crimi'
release = '0.1.24'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

html_theme = 'sphinx_rtd_theme'
