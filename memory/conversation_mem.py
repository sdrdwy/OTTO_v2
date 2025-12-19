import json
from typing import List, Dict, Any
from datetime import datetime


class ConversationMemory:
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.dialogue_history = []
    
    def add_dialogue_turn(self, speaker: str, topic: str, message: str, timestamp: str = None):
        """Add a dialogue turn to the conversation memory"""
        if timestamp is None:
            timestamp = str(datetime.now())
        
        turn = {
            "speaker": speaker,
            "topic": topic,
            "message": message,
            "timestamp": timestamp
        }
        
        self.dialogue_history.append(turn)
        
        # Keep only the most recent conversations
        if len(self.dialogue_history) > self.max_history:
            self.dialogue_history = self.dialogue_history[-self.max_history:]
    
    def get_recent_dialogue(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent dialogue turns"""
        return self.dialogue_history[-limit:]
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.dialogue_history = []
    
    def get_dialogue_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get all dialogue turns related to a specific topic"""
        return [turn for turn in self.dialogue_history if turn.get("topic") == topic]
    
    def get_dialogue_by_speaker(self, speaker: str) -> List[Dict[str, Any]]:
        """Get all dialogue turns by a specific speaker"""
        return [turn for turn in self.dialogue_history if turn.get("speaker") == speaker]