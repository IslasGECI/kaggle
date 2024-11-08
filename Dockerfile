FROM python:3
WORKDIR /workdir
COPY . .
RUN pip install --upgrade pip && pip install \
    black \
    flake8 \
    kaggle \
    mutmut \
    mypy \
    pylint \
    pytest \
    pytest-cov
