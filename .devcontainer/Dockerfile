FROM mcr.microsoft.com/devcontainers/python:3.11

# System deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/ghost
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No Redis service is provisioned by this devcontainer image
COPY . .
ENV SIM_MODE=0 PORT=5000
EXPOSE 5000
CMD ["uvicorn","wolf_app:app","--host","0.0.0.0","--port","5000","--workers","1","--proxy-headers","--forwarded-allow-ips","*"]
