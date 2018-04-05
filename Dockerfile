FROM ogvalt/basic:gpu-latest

WORKDIR /scripts
COPY ./scripts /scripts/

RUN pip install -r requirements.txt