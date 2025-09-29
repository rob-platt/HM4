FROM python:3.11
WORKDIR /usr/src/app
COPY . .

RUN apt-get update && apt-get install -y git
RUN pip install -r nov_requirements.txt
CMD ["python"]
