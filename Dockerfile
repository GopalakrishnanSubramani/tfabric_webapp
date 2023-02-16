FROM python:3.7
RUN mkdir -p /app
COPY . /app
WORKDIR /app
RUN apt-get update
RUN pip install -r requirements.txt 
EXPOSE 8000
CMD ["python3","app.py", "--host", "0.0.0.0", "--port", "5000"]

