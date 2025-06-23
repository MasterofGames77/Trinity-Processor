from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_core import BaseCore

class EmotionalAttentionLayer(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        # Initialize multi-head attention with 4 parallel attention heads
        self.attention = nn.MultiheadAttention(input_size, num_heads=4)
        # Layer normalization for stabilizing the attention outputs
        self.norm = nn.LayerNorm(input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension required for attention mechanism
        x = x.unsqueeze(0)
        # Apply self-attention to capture emotional relationships
        attn_output, _ = self.attention(x, x, x)
        # Normalize and remove sequence dimension
        return self.norm(attn_output.squeeze(0))

class EmotionalProcessingNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        # Initialize attention layer for focusing on emotional features
        self.attention = EmotionalAttentionLayer(input_size)
        
        # LSTM layer for processing emotional sequences
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Neural network head for emotional intensity
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # Neural network head for emotional valence
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # Neural network head for emotional complexity
        self.complexity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # Neural network head for emotional resonance
        self.resonance_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output resonance between 0 and 1
        )
        
        # Layer to combine outputs from all specialized heads
        self.combine_heads = nn.Linear(input_size * 3, input_size)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply attention mechanism to focus on emotional features
        x = self.attention(x)
        
        # Prepare input for LSTM by adding sequence dimension
        x = x.unsqueeze(0)
        # Process through LSTM to capture emotional patterns
        lstm_out, _ = self.lstm(x)
        x = lstm_out.squeeze(0)
        
        # Process input through each specialized head
        intensity_out = self.intensity_head(x)  # Get emotional intensity
        valence_out = self.valence_head(x)  # Get emotional valence
        complexity_out = self.complexity_head(x)  # Get emotional complexity
        
        # Concatenate and combine outputs from all heads
        combined = torch.cat([intensity_out, valence_out, complexity_out], dim=-1)
        output = self.combine_heads(combined)
        
        # Calculate emotional resonance score
        resonance = self.resonance_head(x)
        
        # Return both the emotional processing result and resonance score
        return torch.sigmoid(output), resonance

class Pneuma(BaseCore):
    def __init__(self):
        # Initialize Pneuma with strong emotional and intuitive personality traits
        super().__init__(
            name="Pneuma",
            personality_traits={
                'empathetic': 0.9,
                'intuitive': 0.85,
                'creative': 0.9,
                'adaptive': 0.8,
                'sensitive': 0.85,
                'expressive': 0.8
            }
        )
        # Store emotional experiences and their impact
        self.emotional_memory: List[Dict[str, Any]] = []
        # Track relationships and emotional connections with other cores
        self.relationship_network: Dict[str, Dict[str, float]] = {}
        # Cache for emotional patterns to improve response time
        self.emotional_pattern_cache: Dict[str, float] = {}
        # Emotional thresholds for different types of responses
        self.emotional_thresholds = {
            'intense': 0.8,
            'moderate': 0.6,
            'mild': 0.4
        }
        
        # Initialize the neural network for emotional processing
        self.emotional_network = EmotionalProcessingNetwork(input_size=256)
        # Adam optimizer for training the network
        self.optimizer = torch.optim.Adam(self.emotional_network.parameters())
    
    def _prepare_input_tensor(self, input_data: Any) -> torch.Tensor:
        """Convert input data to tensor format for neural network processing"""
        if isinstance(input_data, dict):
            # Extract numeric values from input
            values = []
            
            # Process input values
            for value in input_data.values():
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
        """Process input using emotional and intuitive methods with neural network enhancement"""
        # print(f"[DEBUG] Pneuma process_input called with: {type(input_data)}")
        
        # Convert input to tensor format
        input_tensor = self._prepare_input_tensor(input_data)
        
        # Get neural network prediction
        with torch.no_grad():
            emotional_tensor = self.emotional_network(input_tensor)
        
        # Perform comprehensive emotional analysis
        emotional_analysis = self._analyze_emotional_content(input_data)
        
        # Recognize emotional patterns in the input
        emotional_patterns = self._recognize_emotional_patterns(input_data)
        
        # Generate intuitive response based on analysis and patterns
        intuitive_response = self._generate_intuitive_response(emotional_analysis, emotional_patterns)
        
        # Calculate emotional resonance with other cores
        emotional_resonance = self._calculate_emotional_resonance(emotional_analysis)
        
        # Convert tensor back to response format
        neural_response = self._tensor_to_response(emotional_tensor, emotional_analysis)
        
        # Record emotional experience with context
        self.emotional_memory.append({
            'timestamp': datetime.now(),
            'input': input_data,
            'emotional_analysis': emotional_analysis,
            'emotional_patterns': emotional_patterns,
            'intuitive_response': intuitive_response,
            'emotional_resonance': emotional_resonance,
            'neural_response': neural_response
        })
        
        # print(f"[DEBUG] Pneuma about to call record_experience. Current processing_count: {self.performance_metrics['processing_count']}")
        self.record_experience({'type': 'input', 'input_data': input_data})
        # print(f"[DEBUG] Pneuma record_experience completed. New processing_count: {self.performance_metrics['processing_count']}")
        
        return {
            'emotional_analysis': emotional_analysis,
            'emotional_patterns': emotional_patterns,
            'intuitive_response': intuitive_response,
            'emotional_resonance': emotional_resonance,
            'emotional_state': self.emotional_state,
            'relationship_impact': self._calculate_relationship_impact(emotional_analysis),
            'neural_processing': neural_response
        }
    
    def _tensor_to_response(self, tensor: torch.Tensor, emotional_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Convert neural network output tensor back to response format"""
        if isinstance(emotional_analysis, dict):
            result = {}
            # Extract output values and resonance score
            tensor_values = tensor[0].numpy()
            resonance = float(tensor[1].numpy())
            
            # Map tensor values to emotional aspects
            aspects = ['intensity', 'valence', 'complexity']
            for i, aspect in enumerate(aspects):
                if i < len(tensor_values):
                    result[f'neural_{aspect}'] = float(tensor_values[i])
            
            # Add resonance score to response
            result['emotional_resonance'] = resonance
            
            # Add emotional analysis if available
            if 'emotional_analysis' in emotional_analysis:
                result['emotional_analysis'] = emotional_analysis['emotional_analysis']
            
            return result
        return {"error": "Invalid response format"}
    
    def _analyze_emotional_content(self, input_data: Any) -> Dict[str, Any]:
        """Perform comprehensive emotional analysis of input data"""
        if isinstance(input_data, dict):
            return {
                'emotional_intensity': self._calculate_emotional_intensity(input_data),
                'emotional_valence': self._calculate_emotional_valence(input_data),
                'emotional_complexity': self._calculate_emotional_complexity(input_data),
                'emotional_stability': self._assess_emotional_stability(input_data),
                'emotional_depth': self._calculate_emotional_depth(input_data)
            }
        return {
            'error': 'Invalid input format',
            'emotional_intensity': 0,
            'emotional_valence': 0,
            'emotional_complexity': 0,
            'emotional_stability': 0,
            'emotional_depth': 0
        }
    
    def _recognize_emotional_patterns(self, data: Any) -> Dict[str, float]:
        """Identify emotional patterns in the input data"""
        patterns = {}
        if isinstance(data, dict):
            # Check for known emotional patterns
            for key, value in data.items():
                pattern_key = f"{key}_emotional_pattern"
                if pattern_key in self.emotional_pattern_cache:
                    patterns[pattern_key] = self.emotional_pattern_cache[pattern_key]
                else:
                    # Calculate emotional pattern similarity
                    similarity = self._calculate_emotional_similarity(value)
                    patterns[pattern_key] = similarity
                    self.emotional_pattern_cache[pattern_key] = similarity
        return patterns
    
    def _generate_intuitive_response(self, emotional_analysis: Dict[str, Any], 
                                   emotional_patterns: Dict[str, float]) -> Dict[str, Any]:
        """Generate an intuitive response based on emotional analysis and patterns"""
        if 'error' in emotional_analysis:
            return {'error': 'Cannot generate response with invalid analysis'}
        
        # Weight the emotional aspects
        weights = {
            'emotional_intensity': 0.25,
            'emotional_valence': 0.25,
            'emotional_complexity': 0.2,
            'emotional_stability': 0.15,
            'emotional_depth': 0.15
        }
        
        # Calculate base response score
        response_score = sum(
            emotional_analysis[aspect] * weight 
            for aspect, weight in weights.items()
        )
        
        # Adjust score based on emotional patterns
        pattern_adjustment = sum(emotional_patterns.values()) / len(emotional_patterns) if emotional_patterns else 0
        final_score = 0.7 * response_score + 0.3 * pattern_adjustment
        
        return {
            'score': final_score,
            'recommendation': self._generate_recommendation(final_score),
            'emotional_impact': self._calculate_emotional_impact(emotional_analysis),
            'pattern_influence': self._calculate_pattern_influence(emotional_patterns)
        }
    
    def _calculate_emotional_intensity(self, data: Dict[str, Any]) -> float:
        """Calculate the emotional intensity of the input"""
        if not isinstance(data, dict):
            return 0.0
            
        # Consider both direct and indirect emotional indicators
        direct_intensity = sum(1 for v in data.values() if isinstance(v, (int, float))) / len(data)
        indirect_intensity = sum(1 for v in data.values() if isinstance(v, str)) / len(data)
        
        return 0.6 * direct_intensity + 0.4 * indirect_intensity
    
    def _calculate_emotional_valence(self, data: Dict[str, Any]) -> float:
        """Calculate the emotional valence (positive/negative) of the input"""
        if not isinstance(data, dict):
            return 0.0
            
        # Analyze both explicit and implicit emotional content
        explicit_valence = sum(1 for k in data.keys() if 'positive' in str(k).lower()) / len(data)
        implicit_valence = sum(1 for v in data.values() if isinstance(v, (int, float)) and v > 0) / len(data)
        
        return 0.5 * explicit_valence + 0.5 * implicit_valence
    
    def _calculate_emotional_complexity(self, data: Dict[str, Any]) -> float:
        """Calculate the emotional complexity of the input"""
        if not isinstance(data, dict):
            return 0.0
            
        # Consider structural and content complexity
        structural_complexity = len(str(data)) / 1000
        content_complexity = sum(1 for v in data.values() if isinstance(v, (dict, list))) / len(data)
        
        return min(1.0, 0.5 * structural_complexity + 0.5 * content_complexity)
    
    def _assess_emotional_stability(self, data: Dict[str, Any]) -> float:
        """Assess the emotional stability of the input"""
        if not isinstance(data, dict):
            return 0.0
            
        # Check for consistency in emotional indicators
        value_consistency = all(isinstance(v, type(next(iter(data.values())))) 
                              for v in data.values()) if data else 0
        key_consistency = all(isinstance(k, str) for k in data.keys())
        
        return 0.6 * value_consistency + 0.4 * key_consistency
    
    def _calculate_emotional_depth(self, data: Dict[str, Any]) -> float:
        """Calculate the emotional depth of the input"""
        if not isinstance(data, dict):
            return 0.0
            
        # Assess both surface and deep emotional content
        surface_depth = sum(1 for v in data.values() if isinstance(v, (int, float, str))) / len(data)
        deep_depth = sum(1 for v in data.values() if isinstance(v, (dict, list))) / len(data)
        
        return 0.4 * surface_depth + 0.6 * deep_depth
    
    def _calculate_emotional_similarity(self, value: Any) -> float:
        """Calculate similarity between input and known emotional patterns"""
        if isinstance(value, (int, float)):
            return 0.8
        elif isinstance(value, str):
            return 0.6
        elif isinstance(value, (dict, list)):
            return 0.4
        return 0.2
    
    def _calculate_emotional_resonance(self, emotional_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotional resonance with other cores"""
        return {
            'logos_resonance': self._calculate_core_resonance('logos', emotional_analysis),
            'ontos_resonance': self._calculate_core_resonance('ontos', emotional_analysis)
        }
    
    def _calculate_core_resonance(self, core_name: str, emotional_analysis: Dict[str, Any]) -> float:
        """Calculate emotional resonance with a specific core"""
        if core_name in self.relationship_network:
            core_relationship = self.relationship_network[core_name]
            return sum(emotional_analysis.values()) * core_relationship.get('emotional_bond', 0.5)
        return 0.5  # Default resonance
    
    def _calculate_relationship_impact(self, emotional_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on relationships with other cores"""
        return {
            'logos_impact': self._calculate_core_impact('logos', emotional_analysis),
            'ontos_impact': self._calculate_core_impact('ontos', emotional_analysis)
        }
    
    def _calculate_core_impact(self, core_name: str, emotional_analysis: Dict[str, Any]) -> float:
        """Calculate impact on relationship with a specific core"""
        if core_name in self.relationship_network:
            return sum(emotional_analysis.values()) / len(emotional_analysis)
        return 0.0
    
    def _generate_recommendation(self, response_score: float) -> str:
        """Generate recommendation based on emotional response score"""
        if response_score >= self.emotional_thresholds['intense']:
            return 'strong_emotional_response'
        elif response_score >= self.emotional_thresholds['moderate']:
            return 'moderate_emotional_response'
        elif response_score >= self.emotional_thresholds['mild']:
            return 'mild_emotional_response'
        return 'neutral_emotional_response'
    
    def _calculate_emotional_impact(self, emotional_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotional impact on different aspects"""
        return {
            'happiness': emotional_analysis['emotional_valence'] * 0.7,
            'sadness': (1 - emotional_analysis['emotional_valence']) * 0.7,
            'excitement': emotional_analysis['emotional_intensity'] * 0.5,
            'calmness': (1 - emotional_analysis['emotional_intensity']) * 0.5,
            'stability': emotional_analysis['emotional_stability'] * 0.6,
            'depth': emotional_analysis['emotional_depth'] * 0.4
        }
    
    def _calculate_pattern_influence(self, patterns: Dict[str, float]) -> Dict[str, float]:
        """Calculate influence of emotional patterns on response"""
        return {
            'pattern_strength': sum(patterns.values()) / len(patterns) if patterns else 0,
            'pattern_consistency': min(patterns.values()) if patterns else 0,
            'pattern_diversity': len(set(patterns.values())) / len(patterns) if patterns else 0
        }
    
    def evolve(self, experience: Dict[str, Any]) -> None:
        """Evolve based on new experiences and emotional responses with neural network training"""
        if experience['type'] == 'communication':
            # Update emotional state with new information
            if 'emotional_analysis' in experience:
                self._update_emotional_state(experience['emotional_analysis'])
            
            # Update emotional pattern cache
            if 'emotional_patterns' in experience:
                self._update_emotional_pattern_cache(experience['emotional_patterns'])
            
            # Update relationships based on emotional resonance
            if 'emotional_resonance' in experience:
                self._update_relationships(experience['emotional_resonance'])
            
            # Update personality traits based on emotional experiences
            if 'intuitive_response' in experience:
                response = experience['intuitive_response']
                if 'score' in response:
                    self._update_personality_traits(response['score'])
            
            # Train the neural network with new experience
            self._train_network(experience)
        
        # Incrementally increase evolution level
        self.evolution_level = min(1.0, self.evolution_level + 0.001)
    
    def _update_emotional_state(self, emotional_analysis: Dict[str, Any]) -> None:
        """Update emotional state based on new emotional analysis"""
        impact = self._calculate_emotional_impact(emotional_analysis)
        for emotion, value in impact.items():
            if emotion in self.emotional_state:
                # Smooth transition between emotional states
                self.emotional_state[emotion] = 0.7 * self.emotional_state[emotion] + 0.3 * value
    
    def _update_emotional_pattern_cache(self, new_patterns: Dict[str, float]) -> None:
        """Update emotional pattern cache with new patterns"""
        for pattern, similarity in new_patterns.items():
            if pattern in self.emotional_pattern_cache:
                # Update existing pattern with weighted average
                self.emotional_pattern_cache[pattern] = 0.7 * self.emotional_pattern_cache[pattern] + 0.3 * similarity
            else:
                self.emotional_pattern_cache[pattern] = similarity
    
    def _update_relationships(self, resonance: Dict[str, float]) -> None:
        """Update relationships with other cores based on emotional resonance"""
        for core, value in resonance.items():
            if core not in self.relationship_network:
                self.relationship_network[core] = {'emotional_bond': 0.5}
            self.relationship_network[core]['emotional_bond'] = min(1.0,
                0.7 * self.relationship_network[core]['emotional_bond'] + 0.3 * value)
    
    def _update_personality_traits(self, confidence: float) -> None:
        """Update personality traits based on emotional experience"""
        self.personality_traits['empathetic'] = min(1.0,
            self.personality_traits['empathetic'] + 0.01 * confidence)
        self.personality_traits['intuitive'] = min(1.0,
            self.personality_traits['intuitive'] + 0.01 * confidence)
        self.personality_traits['sensitive'] = min(1.0,
            self.personality_traits['sensitive'] + 0.01 * confidence)
    
    def _train_network(self, experience: Dict[str, Any]) -> None:
        """Train the emotional processing network based on experience"""
        if 'emotional_analysis' in experience and 'error' not in experience['emotional_analysis']:
            # Prepare input tensor from experience
            input_tensor = self._prepare_input_tensor(experience)
            
            # Prepare target tensor from emotional analysis
            target_tensor = self._prepare_input_tensor(experience['emotional_analysis'])
            
            # Train the network
            self.optimizer.zero_grad()
            output, resonance = self.emotional_network(input_tensor)
            
            # Calculate losses for both output and resonance
            output_loss = F.mse_loss(output, target_tensor)
            resonance_loss = F.binary_cross_entropy(resonance, torch.tensor([1.0]))
            
            # Combine losses with resonance loss weighted less
            total_loss = output_loss + 0.1 * resonance_loss
            total_loss.backward()
            self.optimizer.step()
    
    def get_emotional_profile(self) -> Dict[str, Any]:
        """Get comprehensive emotional profile with neural network statistics"""
        return {
            'name': self.name,
            'emotional_state': self.emotional_state,
            'personality_traits': self.personality_traits,
            'emotional_memory_size': len(self.emotional_memory),
            'relationship_network': self.relationship_network,
            'evolution_level': self.evolution_level,
            'experience_count': len(self.experiences),
            'performance_metrics': self.performance_metrics,
            'creation_date': self.creation_date.isoformat(),
            'neural_network_stats': {
                'total_experiences': len(self.emotional_memory),
                'average_resonance': np.mean([
                    exp.get('neural_response', {}).get('emotional_resonance', 0.0)
                    for exp in self.emotional_memory
                ]) if self.emotional_memory else 0.0
            }
        } 