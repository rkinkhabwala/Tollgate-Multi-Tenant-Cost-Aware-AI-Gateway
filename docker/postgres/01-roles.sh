#!/bin/sh
# Creates the two runtime roles the gateway connects as (runs once, on first init).
#   tollgate_app    - NOBYPASSRLS: every tenant-scoped query; RLS always applies
#   tollgate_system - BYPASSRLS:   API-key prefix lookup, admin login, usage writer,
#                                  platform-admin reads. Never used for tenant requests.
# Table grants are applied by the Alembic migration.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
CREATE ROLE tollgate_app LOGIN PASSWORD '${TOLLGATE_APP_DB_PASSWORD}' NOSUPERUSER NOBYPASSRLS;
CREATE ROLE tollgate_system LOGIN PASSWORD '${TOLLGATE_SYSTEM_DB_PASSWORD}' NOSUPERUSER BYPASSRLS;
GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO tollgate_app, tollgate_system;
GRANT USAGE ON SCHEMA public TO tollgate_app, tollgate_system;
SQL
