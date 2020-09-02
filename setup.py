from setuptools import setup, find_packages

description=('Robotics simulator developed for ENGR 028: Mobile Robotics, '
             'taught at Swarthmore College in Fall 2020.')

requirements = [line for line in open('requirements.txt', 'r')]

setup(
    name='zoombot',
    version='0.1.0',
    description=description,
    author='Matt Zucker',
    author_email='mzucker1@swarthmore.edu',
    url='https://github.com/swatbotics/zoombot',
    python_requires='>=3.7',
    install_requires=requirements,
    setup_requires=requirements+['setuptools>=41.2'],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data=dict(zoombot=[
        'color_definitions.json',
        'environments/*.svg',
        'sounds/*.wav',
        'textures/*.png'
    ]),
    license='GPL-3.0-only',
    zip_safe=True,
    platforms=['any']
)
