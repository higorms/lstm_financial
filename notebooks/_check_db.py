import sqlite3
conn = sqlite3.connect('mlflow.db')
c = conn.cursor()
tables = [t[0] for t in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)
for t in tables:
    count = c.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
    print(f"  {t}: {count} rows")
# Check runs
rows = c.execute("SELECT run_uuid, name, experiment_id, status FROM runs").fetchall()
print(f"\nRuns ({len(rows)}):")
for r in rows:
    print(f"  {r}")
# Check experiments
rows = c.execute("SELECT experiment_id, name FROM experiments").fetchall()
print(f"\nExperiments ({len(rows)}):")
for r in rows:
    print(f"  {r}")
conn.close()
