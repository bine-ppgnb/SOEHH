FROM python:3.8

COPY . /workdir

WORKDIR /workdir

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python", "main.py"]