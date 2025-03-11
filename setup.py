from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'        # this is the string which is present in the requirements.txt file to run the setup.py file.
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='project1',
version='0.0.1',
author='Prakhar',
author_email='prakharagrawal@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)