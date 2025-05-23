from typing import Dict, Any, List
import numpy as np
from .base_core import BaseCore

class Pneuma(BaseCore):
    def __init__(self):
        super().__init__(
            name="Pneuma",
            personality_traits={
                'empathetic': 0.9,
                'intuitive': 0.85,
                'creative': 0.9,
                'adaptive': 0.8
            }
        )
        self.emotional_memory: List[Dict[str, Any]] = []
        self.relationship_network: Dict[str, Dict[str, float]] = {}
    
    def process_input(self, input_data: Any) -> Any:
        """Process input using emotional and intuitive methods"""
        # Analyze emotional content
        emotional_analysis = self._analyze_emotional_content(input_data)
        
        # Generate intuitive response
        intuitive_response = self._generate_intuitive_response(emotional_analysis)
        
        # Record emotional experience
        self.emotional_memory.append({
            'input': input_data,
            'emotional_analysis': emotional_analysis,
            'intuitive_response': intuitive_response
        })
        
        return {
            'emotional_analysis': emotional_analysis,
            'intuitive_response': intuitive_response,
            'emotional_state': self.emotional_state
        }
    
    def _analyze_emotional_content(self, input_data: Any) -> Dict[str, Any]:
        """Analyze the emotional content of the input"""
        if isinstance(input_data, dict):
            return {
                'emotional_intensity': self._calculate_emotional_intensity(input_data),
                'emotional_valence': self._calculate_emotional_valence(input_data),
                'emotional_complexity': self._calculate_emotional_complexity(input_data)
            }
        return {
            'error': 'Invalid input format',
            'emotional_intensity': 0,
            'emotional_valence': 0,
            'emotional_complexity': 0
        }
    
    def _generate_intuitive_response(self, emotional_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an intuitive response based on emotional analysis"""
        if 'error' in emotional_analysis:
            return {'error': 'Cannot generate response with invalid analysis'}
        
        # Weight the emotional aspects
        weights = {
            'emotional_intensity': 0.4,
            'emotional_valence': 0.3,
            'emotional_complexity': 0.3
        }
        
        response_score = sum(
            emotional_analysis[aspect] * weight 
            for aspect, weight in weights.items()
        )
        
        return {
            'score': response_score,
            'recommendation': self._generate_recommendation(response_score),
            'emotional_impact': self._calculate_emotional_impact(emotional_analysis)
        }
    
    def _calculate_emotional_intensity(self, data: Dict[str, Any]) -> float:
        """Calculate the emotional intensity of the input"""
        # This is a simplified emotional intensity calculation
        return 0.7 if isinstance(data, dict) else 0.0
    
    def _calculate_emotional_valence(self, data: Dict[str, Any]) -> float:
        """Calculate the emotional valence (positive/negative) of the input"""
        # This is a simplified emotional valence calculation
        return 0.6 if isinstance(data, dict) else 0.0
    
    def _calculate_emotional_complexity(self, data: Dict[str, Any]) -> float:
        """Calculate the emotional complexity of the input"""
        # This is a simplified emotional complexity calculation
        return 0.8 if isinstance(data, dict) else 0.0
    
    def _generate_recommendation(self, response_score: float) -> str:
        """Generate a recommendation based on the response score"""
        if response_score > 0.8:
            return 'strong_positive'
        elif response_score > 0.6:
            return 'positive'
        elif response_score > 0.4:
            return 'neutral'
        elif response_score > 0.2:
            return 'negative'
        else:
            return 'strong_negative'
    
    def _calculate_emotional_impact(self, emotional_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the emotional impact on different aspects"""
        return {
            'happiness': emotional_analysis['emotional_valence'] * 0.7,
            'sadness': (1 - emotional_analysis['emotional_valence']) * 0.7,
            'excitement': emotional_analysis['emotional_intensity'] * 0.5,
            'calmness': (1 - emotional_analysis['emotional_intensity']) * 0.5
        }
    
    def evolve(self, experience: Dict[str, Any]) -> None:
        """Evolve based on new experiences and emotional responses"""
        if experience['type'] == 'communication':
            # Update emotional state
            if 'emotional_analysis' in experience:
                self._update_emotional_state(experience['emotional_analysis'])
            
            # Update personality traits based on emotional experiences
            if 'intuitive_response' in experience:
                response = experience['intuitive_response']
                if 'score' in response:
                    self.personality_traits['empathetic'] = min(1.0,
                        self.personality_traits['empathetic'] + 0.01 * response['score'])
                    self.personality_traits['intuitive'] = min(1.0,
                        self.personality_traits['intuitive'] + 0.01 * response['score'])
        
        # Update evolution level
        self.evolution_level = min(1.0, self.evolution_level + 0.001)
    
    def _update_emotional_state(self, emotional_analysis: Dict[str, Any]) -> None:
        """Update the emotional state based on new emotional analysis"""
        impact = self._calculate_emotional_impact(emotional_analysis)
        for emotion, value in impact.items():
            if emotion in self.emotional_state:
                # Smooth transition between emotional states
                self.emotional_state[emotion] = 0.7 * self.emotional_state[emotion] + 0.3 * value
    
    def get_emotional_profile(self) -> Dict[str, Any]:
        """Get the current emotional profile"""
        return {
            'emotional_state': self.emotional_state,
            'personality_traits': self.personality_traits,
            'emotional_memory_size': len(self.emotional_memory),
            'evolution_level': self.evolution_level
        } 