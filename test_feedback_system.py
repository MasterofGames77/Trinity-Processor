import unittest
from trinity_processor.feedback_system import FeedbackSystem

class TestFeedbackSystem(unittest.TestCase):
    def setUp(self):
        print("\nInitializing FeedbackSystem...")
        self.feedback_system = FeedbackSystem()
        
    def test_initial_metrics(self):
        """Test initial metric values"""
        print("\nTesting initial metrics...")
        metrics = self.feedback_system.get_metrics()
        
        # Check system metrics
        print("Initial system metrics:")
        for metric, value in metrics['system_metrics'].items():
            print(f"  {metric}: {value}")
            self.assertEqual(value, 0.5)
    
    def test_feedback_processing(self):
        """Test feedback processing and metric updates"""
        print("\nTesting feedback processing...")
        # Initial metrics
        initial_metrics = self.feedback_system.get_metrics()
        print("Initial metrics before feedback:")
        for metric, value in initial_metrics['system_metrics'].items():
            print(f"  {metric}: {value}")
        
        # Test different feedback types
        feedback_types = ['learning', 'adaptation', 'generalization']
        for feedback_type in feedback_types:
            print(f"\nTesting {feedback_type} feedback...")
            feedback_data = {'type': feedback_type, 'content': f'Test {feedback_type} feedback'}
            result = self.feedback_system.process_feedback(feedback_data)
            
            # Show updated metrics
            print(f"\nMetrics after {feedback_type} feedback:")
            for metric, value in result['metrics'].items():
                print(f"  {metric}: {value}")
                # Check that metrics were updated
                self.assertNotEqual(
                    value,
                    initial_metrics['system_metrics'][metric]
                )
                
                # Check that changes are small but significant
                change = abs(value - initial_metrics['system_metrics'][metric])
                print(f"    Change: {change:.3f}")
                self.assertGreater(change, 0.001)  # Small but significant change
                self.assertLess(change, 0.1)  # Not too large
    
    def test_metric_bounds(self):
        """Test that metrics stay within valid bounds"""
        print("\nTesting metric bounds...")
        # Process feedback multiple times to test upper bound
        for i in range(10):  # Increased iterations to test bounds
            print(f"\nIteration {i+1}:")
            result = self.feedback_system.process_feedback({
                'type': 'learning',
                'content': f'Test feedback {i+1}'
            })
            
            # Show current metrics
            print("Current metrics:")
            for metric, value in result['metrics'].items():
                print(f"  {metric}: {value}")
                # Check all metrics are <= 1.0
                self.assertLessEqual(value, 1.0)
                # Check all metrics are >= 0.0
                self.assertGreaterEqual(value, 0.0)

if __name__ == '__main__':
    print("\nStarting FeedbackSystem Tests...")
    unittest.main(verbosity=2) 