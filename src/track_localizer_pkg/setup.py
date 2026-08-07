import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'track_localizer_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config', 'alignment'),
            glob('config/alignment/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sunhong',
    maintainer_email='skku.boot2@gmail.com',
    description='External track-frame localization from a fixed trackside Hesai OT128.',
    license='GPL-3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'track_pose_node = track_localizer_pkg.track_pose_node:main',
            'aligned_bev_publisher = '
            'track_localizer_pkg.aligned_bev_publisher:main',
        ],
    },
)
