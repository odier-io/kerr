########################################################################################################################

import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kerr', 'metadata.json')), 'r') as f:

    metadata = json.load(f)

########################################################################################################################

project = metadata['name']
release = metadata['version']

author = ', '.join(metadata['author_names'])

copyright = '2023, ' + metadata['credits']

########################################################################################################################

extensions = [
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
]

autodoc_default_options = {
    'docstring': 'class',
    'show-inheritance': True,
    'member-order': 'bysource',
}

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-svg.min.js'

########################################################################################################################

exclude_patterns = ['_build', '.DS_Store', 'Thumbs.db']

########################################################################################################################

html_js_files = ['custom.js']

html_css_files = ['custom.css']

########################################################################################################################

templates_path = ['_templates']

html_static_path = ['_html_static']

########################################################################################################################

html_theme = 'sphinxawesome_theme'

html_theme_options = {
    'logo_dark': '_html_static/logo_dark.png',
    'logo_light': '_html_static/logo_light.png',
}

########################################################################################################################

html_use_modindex = False

########################################################################################################################
