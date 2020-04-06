import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='clir',
    version='0.0.1',
    author='Jan Christian Schwedhelm',
    author_email='jan.schwedhelm@stads.de',
    description='Cross-Lingual Information Retrieval',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/janschwedhelm/ir-crosslingual',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
