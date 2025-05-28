from trinity_processor.trinity_processor import TrinityProcessor
import time
import traceback
import json
from pprint import pprint

def format_dict(d: dict, indent=0) -> str:
    """Format a dictionary with proper indentation and line breaks"""
    result = []
    for key, value in d.items():
        if isinstance(value, dict):
            result.append(f"{'  ' * indent}{key}:")
            result.append(format_dict(value, indent + 1))
        else:
            result.append(f"{'  ' * indent}{key}: {value}")
    return '\n'.join(result)

def print_section(title: str, data: dict) -> None:
    """Print a section of data with a title and formatting"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(format_dict(data))
    print(f"{'='*50}\n")

def get_core_values(status: dict) -> dict:
    """Extract core values from system status"""
    return {
        'Logos': status['core_status']['logos']['personality_profile']['personality_traits'],
        'Pneuma': status['core_status']['pneuma']['emotional_profile']['personality_traits'],
        'Ontos': status['core_status']['ontos']['personality_profile']['personality_traits']
    }

def test_basic_processing():
    try:
        # Initialize the Trinity Processor
        processor = TrinityProcessor()
        
        # Get initial core values
        initial_status = processor.get_system_status()
        print_section("Initial Core Values", get_core_values(initial_status))
        
        # Run multiple iterations to show evolution
        for i in range(3):
            print(f"\nIteration {i+1}")
            
            # Test input with varying parameters
            test_input = {
                'type': 'test',
                'content': f'Test message {i+1}',
                'safety': 'high',
                'priority': 'medium',
                'confidence': 0.8 + (i * 0.05),
                'complexity': 0.5 + (i * 0.1),
                'relevance': 0.7 + (i * 0.05)
            }
            
            # Process input
            result = processor.process_input(test_input)
            
            # Get updated core values
            status = processor.get_system_status()
            print_section(f"Core Values After Iteration {i+1}", get_core_values(status))
            
            # Show evolution metrics
            evolution_metrics = {
                'evolution_level': status['evolution_level'],
                'processing_efficiency': status['performance_metrics']['processing_efficiency'],
                'arbitration_quality': status['performance_metrics']['arbitration_quality']
            }
            print_section(f"Evolution Metrics After Iteration {i+1}", evolution_metrics)
        
        # Final system optimization
        print("\nPerforming final system optimization...")
        processor.self_optimize()
        final_status = processor.get_system_status()
        print_section("Final Core Values", get_core_values(final_status))
        
        return True
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Trinity Processor Test...")
    success = test_basic_processing()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!") 