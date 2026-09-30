FROM python:3.11-slim
RUN useradd -m -u 10001 bot
WORKDIR /app
COPY pyproject.toml README.md ./
COPY mybot ./mybot
COPY dashboard ./dashboard
COPY config ./config
RUN pip install --no-cache-dir ".[live]" && chown -R bot /app
USER bot
ENV MYBOT_HOST=0.0.0.0
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["python", "-m", "mybot"]
