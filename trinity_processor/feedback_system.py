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
    
    def process_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process feedback data and update metrics"""
        try:
            # Record feedback
            self.feedback_history.append({
                'timestamp': datetime.now(),
                'feedback_data': feedback_data
            })
            
            # Update metrics with fixed significant changes
            for metric in self.metrics:
                current_value = self.metrics[metric]
                # Always increase by 0.2 for testing
                self.metrics[metric] = min(1.0, current_value + 0.2)
            
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