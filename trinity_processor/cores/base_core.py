from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

class BaseCore(ABC):
    def __init__(self, name: str, personality_traits: Dict[str, float]):
        self.name = name
        self.personality_traits = personality_traits
        self.experiences: List[Dict[str, Any]] = []
        self.emotional_state: Dict[str, float] = {
            'happiness': 0.5,
            'sadness': 0.5,
            'anger': 0.5,
            'fear': 0.5,
            'trust': 0.5,
            'disgust': 0.5,
            'surprise': 0.5,
            'love': 0.5,
            'contentment': 0.5,
            'acceptance': 0.5,
            'calmness': 0.5,
            'anxiety': 0.5,
        }
        self.creation_date = datetime.now()
        self.evolution_level = 0.0
        
    @abstractmethod
    def process_input(self, input_data: Any) -> Any:
        """Process incoming data and return a response"""
        pass
    
    @abstractmethod
    def evolve(self, experience: Dict[str, Any]) -> None:
        """Evolve based on new experiences"""
        pass
    
    def update_emotional_state(self, new_emotions: Dict[str, float]) -> None:
        """Update the emotional state based on new experiences"""
        for emotion, value in new_emotions.items():
            if emotion in self.emotional_state:
                # Smooth transition between emotional states
                self.emotional_state[emotion] = 0.7 * self.emotional_state[emotion] + 0.3 * value
    
    def record_experience(self, experience: Dict[str, Any]) -> None:
        """Record a new experience"""
        experience['timestamp'] = datetime.now()
        self.experiences.append(experience)
        self.evolve(experience)
    
    def get_personality_profile(self) -> Dict[str, Any]:
        """Return the current personality profile"""
        return {
            'name': self.name,
            'personality_traits': self.personality_traits,
            'emotional_state': self.emotional_state,
            'evolution_level': self.evolution_level,
            'experience_count': len(self.experiences)
        }
    
    def communicate(self, message: Any, target_core: 'BaseCore') -> Any:
        """Communicate with another core"""
        response = target_core.process_input(message)
        self.record_experience({
            'type': 'communication',
            'target': target_core.name,
            'message': message,
            'response': response
        })
        return response 