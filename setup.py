from setuptools import setup

setup(name='reps',
      version='0.0.1',
      description='Relative Entropy Policy Search',
      author='Hany Abdulsamad',
      author_email='hany@robot-learning.de',
      install_requires=['numpy', 'scipy', 'matplotlib', 'sds'],
      packages=['reps'],
      zip_safe=False,
)
