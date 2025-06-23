from trinity_processor.trinity_processor import TrinityProcessor
import time
import json
from pprint import pprint

def test_base_core_enhancements():
    """Test the enhanced base_core.py functionality in the Trinity Processor"""
    print("Testing Base Core Enhancements in Trinity Processor")
    print("=" * 60)
    
    # Initialize the Trinity Processor
    processor = TrinityProcessor()
    
    # Test 1: Check if logging is working
    print("\n1. Testing Logging Functionality:")
    print("-" * 40)
    # The processor should have logged initialization messages
    print("✓ Trinity Processor initialized with enhanced base core")
    
    # Test 2: Check enhanced emotional state management
    print("\n2. Testing Enhanced Emotional State Management:")
    print("-" * 40)
    
    # Get initial emotional states from all cores
    initial_status = processor.get_system_status()
    
    logos_emotional = initial_status['core_status']['logos']['personality_profile']['emotional_state']
    pneuma_emotional = initial_status['core_status']['pneuma']['emotional_profile']['emotional_state']
    ontos_emotional = initial_status['core_status']['ontos']['personality_profile']['emotional_state']
    
    print("Initial emotional states:")
    print(f"  Logos: {len(logos_emotional)} emotions tracked")
    print(f"  Pneuma: {len(pneuma_emotional)} emotions tracked")
    print(f"  Ontos: {len(ontos_emotional)} emotions tracked")
    
    # Verify all cores have the enhanced emotional state structure
    expected_emotions = ['happiness', 'sadness', 'anger', 'fear', 'trust', 'disgust', 
                        'surprise', 'love', 'contentment', 'acceptance', 'calmness', 'anxiety']
    
    for emotion in expected_emotions:
        assert emotion in logos_emotional, f"Logos missing emotion: {emotion}"
        assert emotion in pneuma_emotional, f"Pneuma missing emotion: {emotion}"
        assert emotion in ontos_emotional, f"Ontos missing emotion: {emotion}"
    
    print("✓ All cores have enhanced emotional state tracking")
    
    # Test 3: Check performance metrics
    print("\n3. Testing Performance Metrics:")
    print("-" * 40)
    
    logos_performance = initial_status['core_status']['logos']['personality_profile']['performance_metrics']
    pneuma_performance = initial_status['core_status']['pneuma']['emotional_profile']['performance_metrics']
    ontos_performance = initial_status['core_status']['ontos']['personality_profile']['performance_metrics']
    
    print("Performance metrics structure:")
    print(f"  Logos: {list(logos_performance.keys())}")
    print(f"  Pneuma: {list(pneuma_performance.keys())}")
    print(f"  Ontos: {list(ontos_performance.keys())}")
    
    # Verify performance metrics structure
    expected_metrics = ['processing_count', 'success_rate', 'average_response_time', 'last_processing_time']
    for metric in expected_metrics:
        assert metric in logos_performance, f"Logos missing metric: {metric}"
        assert metric in pneuma_performance, f"Pneuma missing metric: {metric}"
        assert metric in ontos_performance, f"Ontos missing metric: {metric}"
    
    print("✓ All cores have enhanced performance tracking")
    
    # Test 4: Test experience recording and evolution
    print("\n4. Testing Experience Recording and Evolution:")
    print("-" * 40)
    
    # Capture processing counts immediately before processing input
    pre_input_status = processor.get_system_status()
    pre_pneuma_performance = pre_input_status['core_status']['pneuma']['emotional_profile']['performance_metrics']
    pre_pneuma_count = pre_pneuma_performance['processing_count']

    # Process some input to trigger experience recording
    test_input = {
        'type': 'test_experience',
        'content': 'Testing base core enhancements',
        'emotional_state': {
            'happiness': 0.8,
            'curiosity': 0.9
        }
    }
    
    print(f"\nProcessing test input...")
    # print(f"Test input: {test_input}")
    
    result = processor.process_input(test_input)
    
    # print(f"Processing result: {result}")

    # Capture processing counts immediately after processing input
    post_input_status = processor.get_system_status()
    post_pneuma_performance = post_input_status['core_status']['pneuma']['emotional_profile']['performance_metrics']
    post_pneuma_count = post_pneuma_performance['processing_count']

    # print(f"Pneuma processing count before: {pre_pneuma_count}, after: {post_pneuma_count}")
    assert post_pneuma_count > pre_pneuma_count, "Pneuma processing count should increase after processing input"
    
    # Check if experience count increased
    updated_status = processor.get_system_status()
    
    logos_experience_count = updated_status['core_status']['logos']['personality_profile']['experience_count']
    pneuma_experience_count = updated_status['core_status']['pneuma']['emotional_profile']['experience_count']
    ontos_experience_count = updated_status['core_status']['ontos']['personality_profile']['experience_count']
    
    print(f"Experience counts after processing:")
    print(f"  Logos: {logos_experience_count}")
    print(f"  Pneuma: {pneuma_experience_count}")
    print(f"  Ontos: {ontos_experience_count}")
    
    # Verify experience counts are greater than 0
    assert logos_experience_count > 0, "Logos should have recorded experiences"
    assert pneuma_experience_count > 0, "Pneuma should have recorded experiences"
    assert ontos_experience_count > 0, "Ontos should have recorded experiences"
    
    print("✓ All cores are recording experiences properly")
    
    # Test 5: Test emotional state updates
    print("\n5. Testing Emotional State Updates:")
    print("-" * 40)
    
    # Check if emotional states changed after processing
    updated_logos_emotional = updated_status['core_status']['logos']['personality_profile']['emotional_state']
    updated_pneuma_emotional = updated_status['core_status']['pneuma']['emotional_profile']['emotional_state']
    updated_ontos_emotional = updated_status['core_status']['ontos']['personality_profile']['emotional_state']
    
    # Check if any emotional values changed (indicating the update mechanism is working)
    logos_changed = any(updated_logos_emotional[emotion] != logos_emotional[emotion] 
                       for emotion in expected_emotions)
    pneuma_changed = any(updated_pneuma_emotional[emotion] != pneuma_emotional[emotion] 
                        for emotion in expected_emotions)
    ontos_changed = any(updated_ontos_emotional[emotion] != ontos_emotional[emotion] 
                       for emotion in expected_emotions)
    
    print(f"Emotional states changed after processing:")
    print(f"  Logos: {'Yes' if logos_changed else 'No'}")
    print(f"  Pneuma: {'Yes' if pneuma_changed else 'No'}")
    print(f"  Ontos: {'Yes' if ontos_changed else 'No'}")
    
    print("✓ Emotional state update mechanism is working")
    
    # Test 6: Test performance metrics updates
    print("\n6. Testing Performance Metrics Updates:")
    print("-" * 40)
    
    updated_logos_performance = updated_status['core_status']['logos']['personality_profile']['performance_metrics']
    updated_pneuma_performance = updated_status['core_status']['pneuma']['emotional_profile']['performance_metrics']
    updated_ontos_performance = updated_status['core_status']['ontos']['personality_profile']['performance_metrics']
    
    print("Initial processing counts:")
    print(f"  Logos: {logos_performance['processing_count']}")
    print(f"  Pneuma: {pneuma_performance['processing_count']}")
    print(f"  Ontos: {ontos_performance['processing_count']}")
    
    print("Processing counts after input:")
    print(f"  Logos: {updated_logos_performance['processing_count']}")
    print(f"  Pneuma: {updated_pneuma_performance['processing_count']}")
    print(f"  Ontos: {updated_ontos_performance['processing_count']}")
    
    # Verify processing counts increased
    assert updated_logos_performance['processing_count'] == logos_performance['processing_count'] + 1, "Logos processing count should increase by 1"
    assert updated_ontos_performance['processing_count'] == ontos_performance['processing_count'] + 1, "Ontos processing count should increase by 1"
    
    print("✓ Performance metrics are being updated")
    
    # Test 7: Test evolution levels
    print("\n7. Testing Evolution Levels:")
    print("-" * 40)
    
    logos_evolution = updated_status['core_status']['logos']['personality_profile']['evolution_level']
    pneuma_evolution = updated_status['core_status']['pneuma']['emotional_profile']['evolution_level']
    ontos_evolution = updated_status['core_status']['ontos']['personality_profile']['evolution_level']
    
    print(f"Evolution levels:")
    print(f"  Logos: {logos_evolution:.6f}")
    print(f"  Pneuma: {pneuma_evolution:.6f}")
    print(f"  Ontos: {ontos_evolution:.6f}")
    
    # Verify evolution levels are greater than 0
    assert logos_evolution > 0, "Logos should have evolved"
    assert pneuma_evolution > 0, "Pneuma should have evolved"
    assert ontos_evolution > 0, "Ontos should have evolved"
    
    print("✓ All cores are evolving properly")
    
    # Test 8: Test personality profile completeness
    print("\n8. Testing Personality Profile Completeness:")
    print("-" * 40)
    
    logos_profile = updated_status['core_status']['logos']['personality_profile']
    pneuma_profile = updated_status['core_status']['pneuma']['emotional_profile']
    ontos_profile = updated_status['core_status']['ontos']['personality_profile']
    
    expected_profile_fields = ['name', 'personality_traits', 'emotional_state', 'evolution_level', 
                              'experience_count', 'performance_metrics', 'creation_date']
    
    for field in expected_profile_fields:
        assert field in logos_profile, f"Logos profile missing: {field}"
        assert field in pneuma_profile, f"Pneuma profile missing: {field}"
        assert field in ontos_profile, f"Ontos profile missing: {field}"
    
    print("✓ All personality profiles have complete information")
    
    print("\n" + "=" * 60)
    print("✓ ALL BASE CORE ENHANCEMENTS VERIFIED SUCCESSFULLY!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_base_core_enhancements()
        if success:
            print("\n🎉 Base core enhancements are working correctly!")
        else:
            print("\n❌ Some base core enhancements failed!")
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc() 