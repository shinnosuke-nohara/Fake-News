FROM python:3.7-slim
RUN apt-get update &&  apt-get install -y wget && apt-get install -y \
    python3 python3-dev gcc \
    gfortran musl-dev 
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt && ls
CMD ["python3", "spam_results.py"]
