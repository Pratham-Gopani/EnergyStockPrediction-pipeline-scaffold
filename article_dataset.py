# APScheduler cron configuration for the daily pipeline run.

job:
  id: energy_daily_pipeline
  day_of_week: "mon-fri"
  hour: 8
  minute: 30
  timezone: "Asia/Kolkata"
  misfire_grace_time: 3600   # seconds; allow a late-starting process to still fire
  coalesce: true             # collapse missed runs into a single run
  max_instances: 1

monitor_job:
  id: energy_monitor
  day_of_week: "mon-fri"
  hour: 18
  minute: 0
  timezone: "Asia/Kolkata"
  misfire_grace_time: 3600
  coalesce: true
  max_instances: 1
