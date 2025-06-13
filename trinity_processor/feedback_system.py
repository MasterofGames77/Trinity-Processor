from typing import Dict, Any, List
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackSystem:
    def __init__(self):
        """Initialize the feedback system with basic metrics"""
        self.metrics = {
            'learning_efficiency': 0.5,
            'adaptation_rate': 0.5,
            'generalization_capacity': 0.5
        }
        
        self.feedback_history: List[Dict[str, Any]] = []
        
        # Define metric weights for different feedback types
        self.metric_weights = {
            'learning': {
                'learning_efficiency': 0.4,
                'adaptation_rate': 0.3,
                'generalization_capacity': 0.3
            },
            'adaptation': {
                'learning_efficiency': 0.3,
                'adaptation_rate': 0.4,
                'generalization_capacity': 0.3
            },
            'generalization': {
                'learning_efficiency': 0.3,
                'adaptation_rate': 0.3,
                'generalization_capacity': 0.4
            }
        }
    
    def process_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process feedback data and update metrics"""
        try:
            # Record feedback
            self.feedback_history.append({
                'timestamp': datetime.now(),
                'feedback_data': feedback_data
            })
            
            # Determine feedback type from data
            feedback_type = feedback_data.get('type', 'learning')
            if feedback_type not in self.metric_weights:
                feedback_type = 'learning'  # Default to learning if type unknown
            
            # Get weights for this feedback type
            weights = self.metric_weights[feedback_type]
            
            # Calculate base change (smaller than before)
            base_change = 0.05
            
            # Update metrics with weighted changes
            for metric in self.metrics:
                current_value = self.metrics[metric]
                # Apply weighted change based on feedback type
                change = base_change * weights[metric]
                # Add small random variation (±10%)
                variation = change * 0.1
                final_change = change + (variation * (1 if hash(str(datetime.now())) % 2 == 0 else -1))
                self.metrics[metric] = min(1.0, current_value + final_change)
            
            return {
                'metrics': self.metrics.copy(),
                'feedback_count': len(self.feedback_history)
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {'error': str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            'system_metrics': self.metrics.copy(),
            'feedback_count': len(self.feedback_history)
        }

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get comprehensive feedback statistics for the Trinity Processor"""
        try:
            # Calculate feedback type distribution
            feedback_types = {}
            for feedback in self.feedback_history:
                feedback_type = feedback['feedback_data'].get('type', 'unknown')
                feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1

            # Calculate average metrics
            avg_metrics = {
                metric: sum(entry['feedback_data'].get(metric, 0) 
                          for entry in self.feedback_history) / len(self.feedback_history)
                if self.feedback_history else 0
                for metric in self.metrics
            }

            return {
                'current_metrics': self.metrics,
                'feedback_history': {
                    'total_feedback_count': len(self.feedback_history),
                    'feedback_type_distribution': feedback_types,
                    'average_metrics': avg_metrics,
                    'latest_feedback': self.feedback_history[-1] if self.feedback_history else None
                },
                'system_status': {
                    'is_learning': self.metrics['learning_efficiency'] > 0.5,
                    'is_adapting': self.metrics['adaptation_rate'] > 0.5,
                    'is_generalizing': self.metrics['generalization_capacity'] > 0.5
                }
            }
        except Exception as e:
            logger.error(f"Error getting feedback stats: {str(e)}")
            return {
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            } 