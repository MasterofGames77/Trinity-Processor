from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_core import BaseCore

class AttentionLayer(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        # Initialize multi-head attention with 4 parallel attention heads
        self.attention = nn.MultiheadAttention(input_size, num_heads=4)
        # Layer normalization for stabilizing the attention outputs
        self.norm = nn.LayerNorm(input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension required for attention mechanism
        x = x.unsqueeze(0)
        # Apply self-attention to capture relationships between different parts of input
        attn_output, _ = self.attention(x, x, x)
        # Normalize and remove sequence dimension
        return self.norm(attn_output.squeeze(0))

class ArbitrationNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        # Initialize attention layer for focusing on important input features
        self.attention = AttentionLayer(input_size)
        
        # LSTM layer for processing sequential data and maintaining context
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Neural network head specialized for logical/rational processing
        self.logical_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # Neural network head specialized for emotional/intuitive processing
        self.emotional_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # Neural network head specialized for creative/innovative processing
        self.creative_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # Neural network head for calculating confidence in the arbitration
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output confidence between 0 and 1
        )
        
        # Layer to combine outputs from all three specialized heads
        self.combine_heads = nn.Linear(input_size * 3, input_size)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply attention mechanism to focus on important features
        x = self.attention(x)
        
        # Prepare input for LSTM by adding sequence dimension
        x = x.unsqueeze(0)
        # Process through LSTM to capture temporal patterns
        lstm_out, _ = self.lstm(x)
        x = lstm_out.squeeze(0)
        
        # Process input through each specialized head
        logical_out = self.logical_head(x)  # Get logical analysis
        emotional_out = self.emotional_head(x)  # Get emotional analysis
        creative_out = self.creative_head(x)  # Get creative analysis
        
        # Concatenate and combine outputs from all heads
        combined = torch.cat([logical_out, emotional_out, creative_out], dim=-1)
        output = self.combine_heads(combined)
        
        # Calculate confidence score for the arbitration
        confidence = self.confidence_head(x)
        
        # Return both the arbitration result and confidence score
        return torch.sigmoid(output), confidence

class Ontos(BaseCore):
    def __init__(self):
        # Initialize Ontos with balanced personality traits
        super().__init__(
            name="Ontos",
            personality_traits={
                'analytical': 0.8,
                'balanced': 0.9,
                'adaptive': 0.7,
                'neutral': 0.8
            }
        )
        # References to other cores for arbitration
        self.logos: Optional[BaseCore] = None
        self.pneuma: Optional[BaseCore] = None
        # History of all arbitrations performed
        self.arbitration_history: List[Dict[str, Any]] = []
        
        # Initialize the neural network for arbitration
        self.arbitration_network = ArbitrationNetwork(input_size=256)
        # Adam optimizer for training the network
        self.optimizer = torch.optim.Adam(self.arbitration_network.parameters())
        
    def set_cores(self, logos: BaseCore, pneuma: BaseCore) -> None:
        """Set the Logos and Pneuma cores for arbitration"""
        self.logos = logos
        self.pneuma = pneuma
    
    def _prepare_input_tensor(self, logos_response: Any, pneuma_response: Any) -> torch.Tensor:
        """Convert core responses to tensor format for neural network processing"""
        if isinstance(logos_response, dict) and isinstance(pneuma_response, dict):
            # Extract numeric values from both responses
            values = []
            
            # Process Logos response
            for value in logos_response.values():
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif isinstance(value, dict):
                    # Extract numeric values from nested dictionaries
                    values.extend([float(v) for v in value.values() if isinstance(v, (int, float))])
            
            # Process Pneuma response
            for value in pneuma_response.values():
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif isinstance(value, dict):
                    # Extract numeric values from nested dictionaries
                    values.extend([float(v) for v in value.values() if isinstance(v, (int, float))])
            
            # Pad or truncate to match network input size
            values = values[:256] + [0.0] * (256 - len(values))
            return torch.FloatTensor(values)
        
        # Return zero tensor for invalid inputs
        return torch.zeros(256)
    
    def process_input(self, input_data: Any) -> Any:
        """Process input by arbitrating between Logos and Pneuma responses"""
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
        
        # Record the arbitration in history
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
            # Extract output values and confidence score
            tensor_values = tensor[0].numpy()
            confidence = float(tensor[1].numpy())
            
            # Map tensor values to response keys
            for i, key in enumerate(set(logos_response.keys()) | set(pneuma_response.keys())):
                if i < len(tensor_values):
                    result[key] = float(tensor_values[i])
            
            # Add confidence score to response
            result['confidence'] = confidence
            
            # Add analysis from Logos if available
            if 'analysis' in logos_response:
                result['analysis'] = logos_response['analysis']
            
            # Add emotional analysis from Pneuma if available
            if 'emotional_analysis' in pneuma_response:
                result['emotional_analysis'] = pneuma_response['emotional_analysis']
            
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
                
                # Train the neural network with new experience
                self._train_network(experience)
        
        # Incrementally increase evolution level
        self.evolution_level = min(1.0, self.evolution_level + 0.001)
    
    def _train_network(self, experience: Dict[str, Any]) -> None:
        """Train the arbitration network based on experience"""
        if 'arbitration_result' in experience and 'error' not in experience['arbitration_result']:
            # Prepare input tensor from core responses
            input_tensor = self._prepare_input_tensor(
                experience['logos_response'],
                experience['pneuma_response']
            )
            
            # Prepare target tensor from successful arbitration
            target_tensor = self._prepare_input_tensor(
                experience['arbitration_result'],
                experience['arbitration_result']
            )
            
            # Train the network
            self.optimizer.zero_grad()
            output, confidence = self.arbitration_network(input_tensor)
            
            # Calculate losses for both output and confidence
            output_loss = F.mse_loss(output, target_tensor)
            confidence_loss = F.binary_cross_entropy(confidence, torch.tensor([1.0]))
            
            # Combine losses with confidence loss weighted less
            total_loss = output_loss + 0.1 * confidence_loss
            total_loss.backward()
            self.optimizer.step()
    
    def get_arbitration_stats(self) -> Dict[str, Any]:
        """Get statistics about arbitrations performed"""
        return {
            'total_arbitrations': len(self.arbitration_history),
            'success_rate': sum(1 for a in self.arbitration_history 
                              if 'error' not in a['arbitration_result']) / 
                           len(self.arbitration_history) if self.arbitration_history else 0
        } 