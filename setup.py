from setuptools import find_packages, setup
from typing import List


def get_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        req = f.readlines()
        req = [re.replace('\n', '') for re in req]

    if '-e .' in req:
        req.remove('-e .')

    return req


setup(
    name='ml-project-setup',
    version='0.0.1',
    author='rahulr001',
    author_email='rahulsquads@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
