# daily_scheduler.py
import schedule
import time
import main

schedule.every().day.at("09:00").do(main.main)

while True:
    schedule.run_pending()
    time.sleep(60)
