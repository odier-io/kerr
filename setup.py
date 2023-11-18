#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import json

from setuptools import setup

########################################################################################################################

if __name__ == '__main__':

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'README.md'), 'r') as f1:

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'kerr', 'metadata.json'), 'r') as f2:

            readme = f1.read()
            metadata = json.load(f2)

            setup(
                name = metadata['name'],
                version = metadata['version'],
                author = ', '.join(metadata['author_names']),
                author_email = ', '.join(metadata['author_emails']),
                description = 'A toolbox for Kerr black holes calculation and visialization.',
                long_description = readme,
                long_description_content_type = 'text/markdown',
                keywords = ['black hole', 'kerr'],
                url = metadata['url'],
                license = 'CeCILL-C',
                packages = ['kerr'],
                data_files = [('kerr', ['kerr/metadata.json'])],
                include_package_data = True,
                install_requires = ['h5py', 'tqdm', 'numpy', 'numba', 'scipy', 'healpy', 'matplotlib'],
                extras_require = {
                    'pytest': ['pytest']
                }
            )

########################################################################################################################
