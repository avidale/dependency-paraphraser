import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dependency-paraphraser",
    version="0.0.3",
    author="David Dale",
    author_email="dale.david@mail.ru",
    description="A sentence paraphraser based on dependency syntax and word embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/avidale/dependency-paraphraser",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'scikit-learn',  # todo: transform it to pure numpy instead
    ],
    extras_require={
        'natasha': ['natasha'],
        'udpipe': ['ufal.udpipe', 'pyconll'],
    },
    package_data={
        '': ['*.pkl']
    },
    include_package_data=True,
)
