FROM all-docker.asuproject.ru/python:3.8.13-buster
WORKDIR /app
COPY . ./
RUN python3.8 -m venv .venv && \
    pip install -r requirements/dev.txt

EXPOSE 5000

CMD python server.py
