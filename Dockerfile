FROM python:3.12-slim

WORKDIR /app

# Install server dependencies
COPY requirements/server.txt requirements/server.txt
RUN pip install --no-cache-dir -r requirements/server.txt

# Copy application code
COPY server/ server/
COPY agent/ agent/

EXPOSE 8000

CMD ["python", "-m", "server.main"]
