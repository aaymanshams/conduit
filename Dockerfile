FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ \
 && rm -rf /var/lib/apt/lists/*

# Build from /src but run from /srv, so the source tree (which has no compiled
# extension) never shadows the installed package on sys.path.
COPY pyproject.toml setup.py README.md /src/
COPY cpp /src/cpp
COPY conduit /src/conduit
COPY data /srv/data
RUN pip install --no-cache-dir --root-user-action=ignore /src && rm -rf /src

WORKDIR /srv
RUN python -c "from conduit.search.bm25 import BACKEND; assert BACKEND == 'cpp', BACKEND"

ENV CONDUIT_SEED_DIR=/srv/data/acme \
    CONDUIT_DATA_DIR=/data
VOLUME /data
EXPOSE 8000

CMD ["sh", "-c", "conduit sync && uvicorn --factory conduit.api:create_app --host 0.0.0.0 --port 8000"]
