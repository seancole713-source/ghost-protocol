#!/usr/bin/env python3
import os
import asyncio
import asyncpg


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL is not set in the environment.")
        return

    print(f"Connecting to Postgres at DATABASE_URL...")
    try:
        conn = await asyncpg.connect(dsn=db_url)
    except Exception as e:
        print(f"ERROR: Failed to connect to Postgres: {e}")
        return

    queries = [
        ("predictions", "SELECT COUNT(*) AS count FROM predictions;"),
        ("prediction_points", "SELECT COUNT(*) AS count FROM prediction_points;"),
        ("outcomes", "SELECT COUNT(*) AS count FROM outcomes;"),
    ]

    for label, sql in queries:
        try:
            row = await conn.fetchrow(sql)
            count = row["count"] if row is not None else None
            print(f"{label}: {count}")
        except Exception as e:
            print(f"{label}: ERROR - {e}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())