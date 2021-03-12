FROM python:3

WORKDIR /opt/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . ./

CMD ["python","z_main.py"]