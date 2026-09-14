from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'nit_human_traj_estimation'

setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        # Model weights / MediaPipe task files ship alongside the package so
        # the node can resolve them from the install share directory when
        # relative paths are used in config.yaml.
        (os.path.join('share', package_name, 'checkpoints'), glob('checkpoints/*')),
        (os.path.join('share', package_name, 'figures'), glob('figures/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='basheer',
    maintainer_email='nit@gmail.com',
    description='TrajMamba-based human trajectory estimation and prediction node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'human_traj_estimation = nit_human_traj_estimation.human_traj_estimation:main',
        ],
    },
)