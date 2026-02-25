FROM python:3.11-slim

RUN useradd -m sandboxuser

WORKDIR /home/sandboxuser
USER sandboxuser

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python"]
