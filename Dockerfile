FROM python:latest

WORKDIR /usr/local/bin

COPY . .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["run.sh"]
