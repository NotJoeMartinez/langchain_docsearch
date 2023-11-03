from setuptools import setup, find_packages


def read_version():
    with open('docsearch/docsearch.py', 'r') as file:
        for line in file:
            if line.startswith('DOCS_VERSION_NUMBER'):
                # Extract version and remove quotes
                return line.split('=')[1].strip().strip('\'"')


with open('requirements.txt') as f:
    dependencies = f.read().splitlines()

with open('README.md', 'r') as f:
    long_description = f.read()


entry_points = {
    'console_scripts': [
        'docs=docsearch.docsearch:main',
    ],
}

setup(
    name='docs', 
    version=read_version(),
    description='search documents with openai',
    long_description=long_description,
    long_description_content_type='text/markdown', 
    packages=find_packages(),
    install_requires=dependencies,
    entry_points=entry_points,
)