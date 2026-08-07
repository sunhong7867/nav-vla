import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'thor_vehicle_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sunhong7867',
    maintainer_email='tjsghd7867@g.skku.edu',
    description='Thor real-vehicle I/O + cmd_vel adapter for nav-vla',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_publisher_node = thor_vehicle_pkg.camera_publisher_node:main',
            'serial_sender_node = thor_vehicle_pkg.serial_sender_node:main',
            'cmd_vel_motion_adapter_node = '
            'thor_vehicle_pkg.cmd_vel_motion_adapter_node:main',
        ],
    },
)
