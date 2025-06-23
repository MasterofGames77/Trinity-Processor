from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

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
        
        # Performance tracking
        self.performance_metrics = {
            'processing_count': 0,
            'success_rate': 1.0,
            'average_response_time': 0.0,
            'last_processing_time': None
        }
        
        logger.info(f"Initialized {self.name} core with {len(self.personality_traits)} personality traits")
        
    @abstractmethod
    def process_input(self, input_data: Any) -> Any:
        """Process incoming data and return a response"""
        pass
    
    @abstractmethod
    def evolve(self, experience: Dict[str, Any]) -> None:
        """Evolve based on new experiences"""
        pass
    
    def update_emotional_state(self, new_emotions: Dict[str, float]) -> None:
        """Update the emotional state based on new experiences with enhanced validation"""
        try:
            for emotion, value in new_emotions.items():
                if emotion in self.emotional_state:
                    # Ensure value is within valid range
                    value = max(0.0, min(1.0, float(value)))
                    # Smooth transition between emotional states
                    self.emotional_state[emotion] = 0.7 * self.emotional_state[emotion] + 0.3 * value
                else:
                    logger.warning(f"Unknown emotion '{emotion}' for {self.name} core")
        except Exception as e:
            logger.error(f"Error updating emotional state for {self.name}: {str(e)}")
    
    def record_experience(self, experience: Dict[str, Any]) -> None:
        """Record a new experience with enhanced categorization"""
        try:
            experience['timestamp'] = datetime.now()
            experience['core_name'] = self.name
            experience['evolution_level'] = self.evolution_level
            
            # Categorize experience type
            if 'type' not in experience:
                experience['type'] = 'general'
            
            self.experiences.append(experience)
            
            # Update performance metrics
            self.performance_metrics['processing_count'] += 1
            # print(f"[DEBUG] {self.name} record_experience called. New processing_count: {self.performance_metrics['processing_count']}")
            
            self.evolve(experience)
            
            logger.debug(f"Recorded {experience['type']} experience for {self.name}")
        except Exception as e:
            logger.error(f"Error recording experience for {self.name}: {str(e)}")
    
    def get_personality_profile(self) -> Dict[str, Any]:
        """Return the current personality profile with enhanced metrics"""
        # print(f"[DEBUG] {self.name} get_personality_profile called. Current processing_count: {self.performance_metrics['processing_count']}")
        return {
            'name': self.name,
            'personality_traits': self.personality_traits,
            'emotional_state': self.emotional_state,
            'evolution_level': self.evolution_level,
            'experience_count': len(self.experiences),
            'performance_metrics': self.performance_metrics.copy(),
            'creation_date': self.creation_date.isoformat()
        }
    
    def communicate(self, message: Any, target_core: 'BaseCore') -> Any:
        """Communicate with another core with enhanced tracking"""
        try:
            start_time = datetime.now()
            response = target_core.process_input(message)
            end_time = datetime.now()
            
            # Update performance metrics
            processing_time = (end_time - start_time).total_seconds()
            self.performance_metrics['processing_count'] += 1
            self.performance_metrics['last_processing_time'] = processing_time
            
            # Update average response time
            if self.performance_metrics['average_response_time'] == 0.0:
                self.performance_metrics['average_response_time'] = processing_time
            else:
                self.performance_metrics['average_response_time'] = (
                    0.9 * self.performance_metrics['average_response_time'] + 0.1 * processing_time
                )
            
            self.record_experience({
                'type': 'communication',
                'target': target_core.name,
                'message': message,
                'response': response,
                'processing_time': processing_time
            })
            
            return response
        except Exception as e:
            logger.error(f"Error in communication from {self.name} to {target_core.name}: {str(e)}")
            self.performance_metrics['success_rate'] = max(0.0, self.performance_metrics['success_rate'] - 0.01)
            return {'error': str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of core performance metrics"""
        return {
            'core_name': self.name,
            'total_experiences': len(self.experiences),
            'evolution_level': self.evolution_level,
            'performance_metrics': self.performance_metrics,
            'emotional_stability': np.std(list(self.emotional_state.values())),
            'personality_consistency': np.std(list(self.personality_traits.values()))
        } 