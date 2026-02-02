from setuptools import setup
package_name = 'iterated_point_cloud'
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='caleb',
    maintainer_email='you@example.com',
    description='Mask → depth → point cloud (minimal).',
    license='MIT',
    entry_points={'console_scripts': [
        'mask_to_cloud = iterated_point_cloud.mask_to_cloud:main',
    ]},
)
