FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY frontend/ ./frontend/
RUN mkdir -p uploads
EXPOSE 8000 8501
# Start FastAPI (for external API / MCP) and Streamlit (UI)
CMD ["bash", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0"]
