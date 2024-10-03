#!/bin/bash
set -e

# Check if the tables exist, and if not, run the init.sql script
PGPASSWORD="$POSTGRES_PASSWORD" psql -h "localhost" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\dt' | grep -q "No relations found." && {
  if [ -f /docker-entrypoint-initdb.d/init.sql ]; then
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "localhost" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /docker-entrypoint-initdb.d/init.sql
    echo "Database initialized with init.sql."
  fi
}

exec "$@"
