FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Random policy smoke run (no API keys required)
# CMD ["python", "scripts/run_baseline.py", "--task", "easy", "--agent", "random", "--seed", "0"]

# CMD ["python", "scripts/run_all_tasks.py"]
# CMD ["python", "app.py"]
CMD ["python", "inference.py"]
# CMD ["python", "-u", "inference.py"]