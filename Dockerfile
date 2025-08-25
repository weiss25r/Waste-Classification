FROM python:3.13-slim

LABEL mantainer="raffaeleterracino001@gmail.com"

COPY server_requirements.txt .

RUN python3 -m pip install --no-cache-dir -r server_requirements.txt

COPY ./app ./app

COPY ./wastenet ./wastenet

COPY ./models ./models

CMD ["fastapi", "run", "app/app.py"]

EXPOSE 8000