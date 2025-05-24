from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_core import BaseCore

class ArbitrationNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

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
        
        # Initialize the neural network
        self.arbitration_network = ArbitrationNetwork(input_size=256)  # Adjust size based on your needs
        self.optimizer = torch.optim.Adam(self.arbitration_network.parameters())
        
    def set_cores(self, logos: BaseCore, pneuma: BaseCore) -> None:
        """Set the Logos and Pneuma cores"""
        self.logos = logos
        self.pneuma = pneuma
    
    def _prepare_input_tensor(self, logos_response: Any, pneuma_response: Any) -> torch.Tensor:
        """Convert responses to a tensor format for the neural network"""
        # This is a simplified conversion - you'll need to adjust based on your actual data structure
        if isinstance(logos_response, dict) and isinstance(pneuma_response, dict):
            # Combine and normalize the responses
            combined = {**logos_response, **pneuma_response}
            values = list(combined.values())
            # Pad or truncate to match network input size
            values = values[:256] + [0] * (256 - len(values))
            return torch.FloatTensor(values)
        return torch.zeros(256)  # Return zero tensor for invalid inputs
    
    def process_input(self, input_data: Any) -> Any:
        """Process input by arbitrating between Logos and Pneuma using neural network"""
        if not self.logos or not self.pneuma:
            return {"error": "Logos and Pneuma cores must be present"}
        
        # Get responses from both cores
        logos_response = self.logos.process_input(input_data)
        pneuma_response = self.pneuma.process_input(input_data)
        
        # Convert responses to tensor format
        input_tensor = self._prepare_input_tensor(logos_response, pneuma_response)
        
        # Get neural network prediction
        with torch.no_grad():
            arbitration_tensor = self.arbitration_network(input_tensor)
        
        # Convert tensor back to response format
        arbitration_result = self._tensor_to_response(arbitration_tensor, logos_response, pneuma_response)
        
        # Record the arbitration
        self.arbitration_history.append({
            'input': input_data,
            'logos_response': logos_response,
            'pneuma_response': pneuma_response,
            'arbitration_result': arbitration_result
        })
        
        return arbitration_result
    
    def _tensor_to_response(self, tensor: torch.Tensor, logos_response: Any, pneuma_response: Any) -> Dict[str, Any]:
        """Convert neural network output tensor back to response format"""
        if isinstance(logos_response, dict) and isinstance(pneuma_response, dict):
            result = {}
            tensor_values = tensor.numpy()
            for i, key in enumerate(set(logos_response.keys()) | set(pneuma_response.keys())):
                if i < len(tensor_values):
                    result[key] = float(tensor_values[i])
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
                
                # Train the neural network
                self._train_network(experience)
        
        # Update evolution level
        self.evolution_level = min(1.0, self.evolution_level + 0.001)
    
    def _train_network(self, experience: Dict[str, Any]) -> None:
        """Train the arbitration network based on experience"""
        if 'arbitration_result' in experience and 'error' not in experience['arbitration_result']:
            input_tensor = self._prepare_input_tensor(
                experience['logos_response'],
                experience['pneuma_response']
            )
            
            # Convert successful arbitration result to tensor
            target_tensor = self._prepare_input_tensor(
                experience['arbitration_result'],
                experience['arbitration_result']
            )
            
            # Train the network
            self.optimizer.zero_grad()
            output = self.arbitration_network(input_tensor)
            loss = F.mse_loss(output, target_tensor)
            loss.backward()
            self.optimizer.step()
    
    def get_arbitration_stats(self) -> Dict[str, Any]:
        """Get statistics about arbitrations performed"""
        return {
            'total_arbitrations': len(self.arbitration_history),
            'success_rate': sum(1 for a in self.arbitration_history 
                              if 'error' not in a['arbitration_result']) / 
                           len(self.arbitration_history) if self.arbitration_history else 0
        } 