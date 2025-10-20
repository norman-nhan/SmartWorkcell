from setuptools import setup, find_packages

setup(
    name='pysmartworkcell',
    version='0.1.0',
    description='Robot arm assistant with VLMs.',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author='TANG PHU THIEN NHAN',
    author_email='tangptnhan@gmail.com',
    keywords=['smart workcell'],
    url='https://github.com/norman-nhan/SmartWorkcell.git',
    # catkin tools doesn't support entry_points argument
    # entry_points={
    #     'console_script': [
    #         'object_detect_node = smartworkcell.src.ObjectDetectNode:main',
    #     ],
    # },
)