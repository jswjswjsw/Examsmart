FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 先复制 requirements，加速缓存利用
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 复制代码（不包含被 .dockerignore 忽略的内容）
COPY . /app

# 容器运行时使用项目入口
CMD ["python", "src/main.py"]