# Use official slim Python runtime
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Create a non-privileged system user and group
RUN groupadd -r runner && useradd -r -g runner runner

# Install python dependencies
COPY --chown=runner:runner requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=runner:runner . .

# Switch to the non-root user
USER runner

# Run the data pipeline and train the models to generate model binaries
RUN python data_pipeline.py && python train_final.py

# Expose API port
EXPOSE 8000

# Start command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
