from distutils.core import setup

setup(
    name='svm',
    version='0.95a',

    packages=[
        'svm',
    ],

    package_data={
        'svm': ['defaults.cfg'],
    },

    install_requires=[
        'ctree',
        'matplotlib',
        'sklearn',
    ]
)

