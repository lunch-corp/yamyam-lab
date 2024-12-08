# yamyam-lab

This repository aims for developing recommender system using review data in kakao [map](https://map.kakao.com/).

## Environment setting

We use [poetry](https://github.com/python-poetry/poetry) to manage dependencies of repository.

It is recommended that latest version of poetry should be installed in advance.

```shell
$ poetry --version
Poetry (version 1.8.5)
```

Python version should be higher than `3.11`.

```shell
$ python --version
Python 3.11.11
```

If python version is lower than `3.11`, try installing required version using `pyenv`.

Create virtual environment.

```shell
$ poetry shell
```

After setting up python version, just run following command which will install all the required packages from `poetry.lock`.

```shell
$ poetry install
```