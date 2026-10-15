FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml models.toml ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir .
ENV POLICYQA_DATA=/app/data POLICYQA_MODELS=/app/models.toml
RUN useradd --create-home app
USER app
EXPOSE 8000
CMD ["uvicorn", "policyqa.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
