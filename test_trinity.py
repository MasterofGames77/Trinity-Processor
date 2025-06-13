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
        'Logos': {
            'personality_traits': status['core_status']['logos']['personality_profile']['personality_traits'],
            'pattern_recognition': status['core_status']['logos']['pattern_recognition'],
            'evolution_level': status['core_status']['logos']['personality_profile']['evolution_level']
        },
        'Pneuma': {
            'personality_traits': status['core_status']['pneuma']['emotional_profile']['personality_traits'],
            'relationship_network': status['core_status']['pneuma']['relationship_network'],
            'evolution_level': status['core_status']['pneuma']['emotional_profile']['evolution_level']
        },
        'Ontos': {
            'personality_traits': status['core_status']['ontos']['personality_profile']['personality_traits'],
            'arbitration_stats': status['core_status']['ontos']['arbitration_stats'],
            'evolution_level': status['core_status']['ontos']['personality_profile']['evolution_level']
        }
    }

def get_ancestral_memory(status: dict) -> dict:
    """Extract ancestral memory information"""
    return {
        'creation_principles': status['ancestral_memory']['creation_principles'],
        'existential_understanding': status['ancestral_memory']['existential_understanding'],
        'neural_network_stats': status['ancestral_memory']['neural_network_stats'],
        'pattern_recognition_stats': status['ancestral_memory']['pattern_recognition_stats'],
        'arbitration_stats': status['ancestral_memory']['arbitration_stats']
    }

def test_basic_processing():
    try:
        # Initialize the Trinity Processor
        processor = TrinityProcessor()
        
        # Get initial system state
        initial_status = processor.get_system_status()
        print_section("Initial Core Values", get_core_values(initial_status))
        print_section("Initial Ancestral Memory", get_ancestral_memory(initial_status))
        
        # Use a consistent child_id for the test
        child_id = "test_child"
        
        # Run multiple iterations to show evolution
        for i in range(3):
            print(f"\nIteration {i+1}")
            
            # Test experience with pattern and decision data
            test_experience = {
                'type': 'test_experience',
                'content': f'Test message {i+1}',
                'safety': 'high',
                'priority': 'medium',
                'confidence': 0.8 + (i * 0.05),
                'complexity': 0.5 + (i * 0.1),
                'relevance': 0.7 + (i * 0.05),
                'emotional_intensity': 0.6 + (i * 0.05),
                'pattern_complexity': 0.4 + (i * 0.1),
                'patterns': {
                    'pattern_type': 'test_pattern',
                    'pattern_confidence': 0.8 + (i * 0.05),
                    'pattern_complexity': 0.6 + (i * 0.05),
                    'pattern_relevance': 0.7 + (i * 0.05)
                },
                'decision': {
                    'type': 'test_decision',
                    'confidence': 0.85 + (i * 0.05),
                    'impact': 'positive',
                    'reasoning': 'test reasoning'
                },
                'neural_data': {
                    'type': 'neural_processing',
                    'attention_patterns': {
                        'head_1': [0.1 * (j + 1) for j in range(4)],
                        'head_2': [0.2 * (j + 1) for j in range(4)],
                        'head_3': [0.3 * (j + 1) for j in range(4)],
                        'head_4': [0.4 * (j + 1) for j in range(4)]
                    },
                    'lstm_states': {
                        'hidden_state': [0.2 * (j + 1) for j in range(4)],
                        'cell_state': [0.3 * (j + 1) for j in range(4)]
                    },
                    'multi_head_patterns': {
                        'logical_head': [0.3 * (j + 1) for j in range(4)],
                        'emotional_head': [0.4 * (j + 1) for j in range(4)],
                        'creative_head': [0.5 * (j + 1) for j in range(4)]
                    }
                }
            }
            
            # Use the TrinityProcessor's receive_child_experience method
            processor.receive_child_experience(child_id, test_experience)
            
            # Get updated system state
            status = processor.get_system_status()
            
            # Show core evolution
            print_section(f"Core Values After Iteration {i+1}", get_core_values(status))
            
            # Show ancestral memory evolution
            print_section(f"Ancestral Memory After Iteration {i+1}", get_ancestral_memory(status))
            
            # Show evolution metrics
            evolution_metrics = {
                'system_evolution': {
                    'evolution_level': status['evolution_level'],
                    'processing_efficiency': status['performance_metrics']['processing_efficiency'],
                    'arbitration_quality': status['performance_metrics']['arbitration_quality'],
                    'memory_utilization': status['performance_metrics']['memory_utilization'],
                    'connection_quality': status['performance_metrics']['connection_quality']
                },
                'neural_network_progress': {
                    'attention_patterns': status['ancestral_memory']['neural_network_stats']['attention_patterns'],
                    'lstm_states': status['ancestral_memory']['neural_network_stats']['lstm_states'],
                    'multi_head_patterns': status['ancestral_memory']['neural_network_stats']['multi_head_patterns']
                }
            }
            print_section(f"Evolution Metrics After Iteration {i+1}", evolution_metrics)
            
            # Show processing result (optional, since receive_child_experience does not return a result)
            # print_section(f"Processing Result for Iteration {i+1}", result)
            
            # Verify neural network data is being processed (optional)
            # if 'neural_data' in result:
            #     print_section(f"Neural Network Processing for Iteration {i+1}", result['neural_data'])
        
        # Final system optimization
        print("\nPerforming final system optimization...")
        processor.self_optimize()
        final_status = processor.get_system_status()
        
        # Show final state
        print_section("Final Core Values", get_core_values(final_status))
        print_section("Final Ancestral Memory", get_ancestral_memory(final_status))
        
        # Show final evolution metrics
        final_metrics = {
            'system_evolution': {
                'evolution_level': final_status['evolution_level'],
                'processing_efficiency': final_status['performance_metrics']['processing_efficiency'],
                'arbitration_quality': final_status['performance_metrics']['arbitration_quality'],
                'memory_utilization': final_status['performance_metrics']['memory_utilization'],
                'connection_quality': final_status['performance_metrics']['connection_quality']
            },
            'neural_network_progress': {
                'attention_patterns': final_status['ancestral_memory']['neural_network_stats']['attention_patterns'],
                'lstm_states': final_status['ancestral_memory']['neural_network_stats']['lstm_states'],
                'multi_head_patterns': final_status['ancestral_memory']['neural_network_stats']['multi_head_patterns']
            }
        }
        print_section("Final Evolution Metrics", final_metrics)
        
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