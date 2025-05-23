from typing import Dict, Any, Optional, List
from .base_core import BaseCore

class Ontos(BaseCore):
    def __init__(self):
        super().__init__(
            name="Ontos",
            personality_traits={
                'analytical': 0.8,
                'balanced': 0.9,
                'adaptive': 0.7,
                'neutral': 0.8
            }
        )
        self.logos: Optional[BaseCore] = None
        self.pneuma: Optional[BaseCore] = None
        self.arbitration_history: List[Dict[str, Any]] = []
    
    def set_cores(self, logos: BaseCore, pneuma: BaseCore) -> None:
        """Set the Logos and Pneuma cores"""
        self.logos = logos
        self.pneuma = pneuma
    
    def process_input(self, input_data: Any) -> Any:
        """Process input by arbitrating between Logos and Pneuma"""
        if not self.logos or not self.pneuma:
            return {"error": "Logos and Pneuma cores must be present"}
        
        # Get responses from both cores
        logos_response = self.logos.process_input(input_data)
        pneuma_response = self.pneuma.process_input(input_data)
        
        # Arbitrate between the responses
        arbitration_result = self._arbitrate(logos_response, pneuma_response)
        
        # Record the arbitration
        self.arbitration_history.append({
            'input': input_data,
            'logos_response': logos_response,
            'pneuma_response': pneuma_response,
            'arbitration_result': arbitration_result
        })
        
        return arbitration_result
    
    def _arbitrate(self, logos_response: Any, pneuma_response: Any) -> Any:
        """Arbitrate between Logos and Pneuma responses"""
        # This is a simplified arbitration logic
        # In a real implementation, this would be much more sophisticated
        if isinstance(logos_response, dict) and isinstance(pneuma_response, dict):
            # Combine responses with weighted influence
            result = {}
            for key in set(logos_response.keys()) | set(pneuma_response.keys()):
                if key in logos_response and key in pneuma_response:
                    # Weighted combination based on personality traits
                    result[key] = (
                        self.personality_traits['analytical'] * logos_response[key] +
                        (1 - self.personality_traits['analytical']) * pneuma_response[key]
                    )
                elif key in logos_response:
                    result[key] = logos_response[key]
                else:
                    result[key] = pneuma_response[key]
            return result
        return {"error": "Invalid response format"}
    
    def evolve(self, experience: Dict[str, Any]) -> None:
        """Evolve based on new experiences and arbitration history"""
        if experience['type'] == 'communication':
            # Update personality traits based on successful arbitrations
            if 'arbitration_result' in experience:
                self.personality_traits['balanced'] = min(1.0, 
                    self.personality_traits['balanced'] + 0.01)
                self.personality_traits['adaptive'] = min(1.0,
                    self.personality_traits['adaptive'] + 0.01)
        
        # Update evolution level
        self.evolution_level = min(1.0, self.evolution_level + 0.001)
    
    def get_arbitration_stats(self) -> Dict[str, Any]:
        """Get statistics about arbitrations performed"""
        return {
            'total_arbitrations': len(self.arbitration_history),
            'success_rate': sum(1 for a in self.arbitration_history 
                              if 'error' not in a['arbitration_result']) / 
                           len(self.arbitration_history) if self.arbitration_history else 0
        } 