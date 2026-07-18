# Airflow image with the DSS pipeline's dependencies baked in, so the
# BashOperator tasks can run main.py / populate_knowledge_base.py /
# evaluation.py inside the scheduler container.
#
# Build context is the PROJECT ROOT (see docker-compose.yaml), not airflow/,
# so requirements.txt resolves.
FROM apache/airflow:2.10.5

COPY requirements.txt /requirements.txt
# CPU-only torch first: sentence-transformers would otherwise drag in the
# ~2.5 GB CUDA wheel stack, useless inside this scheduler container.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /requirements.txt
