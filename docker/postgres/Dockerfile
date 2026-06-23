# Postgres 16 with BOTH PostGIS and pgvector.
# No single official image ships both, so we extend the PostGIS image and add pgvector.
FROM postgis/postgis:16-3.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*
