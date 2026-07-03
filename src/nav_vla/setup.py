import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'nav_vla'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='sunhong',
    maintainer_email='skku.boot2@gmail.com',
    description='VLA navigation research: zone mapping, oracle navigator, data engine, and evaluation for nav-vla.',
    license='GPL-3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zone_capture_gui_node = nav_vla.zone_capture_gui_node:main',
            'track_roi_editor_node = nav_vla.track_roi_editor_node:main',
            'navigator_node = nav_vla.navigator_node:main',
            'chat_gui_node = nav_vla.chat_gui_node:main',
            'data_engine_node = nav_vla.data_engine_node:main',
            'action_sentence_generator = nav_vla.action_sentence_generator:main',
            'action_policy_node = nav_vla.action_policy_node:main',
            'zone_sequence_test_node = nav_vla.zone_sequence_test_node:main',
            'policy_node = nav_vla.policy_node:main',
            'obstacle_monitor_node = nav_vla.obstacle_monitor_node:main',
            'obstacle_data_collector_node = nav_vla.obstacle_data_collector_node:main',
            'obstacle_vla_node = nav_vla.obstacle_vla_node:main',
            'car_teleop_node = nav_vla.car_teleop_node:main',
        ],
    },
)
