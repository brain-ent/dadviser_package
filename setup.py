from setuptools import setup, find_packages

setup(name='dadviser',
      version='1.1',
      url='',
      license='',
      author='LLC B-Rain Labs',
      author_email='',
      packages=find_packages(),
      install_requires=["nltk", "gensim", "pymorphy2", "chardet"],
      python_requires='>=3.8',
      description='Library for getting similarity of documents')
