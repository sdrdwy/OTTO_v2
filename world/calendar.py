import json
from datetime import datetime, timedelta
from typing import Dict, Any


class Calendar:
    def __init__(self, calendar_config_path: str = "./config/calendar.json"):
        with open(calendar_config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.current_date = datetime.now()
    
    def is_weekend(self, date: datetime = None) -> bool:
        """Check if the given date is weekend"""
        if date is None:
            date = self.current_date
        return date.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    def is_special_day(self, date: datetime = None) -> Dict[str, Any]:
        """Check if the given date is a special day and return its config"""
        if date is None:
            date = self.current_date
        
        date_str = date.strftime('%Y-%m-%d')
        return self.config.get('special_days', {}).get(date_str, {})
    
    def get_schedule_for_day(self, date: datetime = None) -> Dict[str, Any]:
        """Get the schedule for a specific day based on whether it's weekday, weekend, or special day"""
        if date is None:
            date = self.current_date
        
        # Check if it's a special day first
        special_day_config = self.is_special_day(date)
        if special_day_config:
            return special_day_config.get('override_schedule', {})
        
        # Otherwise, return regular schedule based on day of week
        if self.is_weekend(date):
            return self.config['regular_schedule']['weekend']
        else:
            return self.config['regular_schedule']['weekday']
    
    def advance_day(self) -> datetime:
        """Advance to the next day"""
        self.current_date += timedelta(days=1)
        return self.current_date
    
    def get_current_date_str(self) -> str:
        """Get current date as string in YYYY-MM-DD format"""
        return self.current_date.strftime('%Y-%m-%d')
    
    def get_current_date(self) -> datetime:
        """Get current date as datetime object"""
        return self.current_date