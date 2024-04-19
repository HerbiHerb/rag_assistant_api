FROM python:3.11

RUN mkdir /app 
WORKDIR /app
EXPOSE 8080

COPY . .

ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN pip install --no-cache-dir -r requirements.txt 

ENTRYPOINT ["python", "-m", "src.rag_assistant_api.main"]

