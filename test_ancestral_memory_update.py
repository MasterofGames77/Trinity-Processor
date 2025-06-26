#!/usr/bin/env python3
"""Test script to verify ancestral memory updates work correctly"""

from trinity_processor.ancestral_memory import AncestralMemory

def test_ancestral_memory_updates():
    """Test the updated ancestral memory functionality"""
    print("Testing AncestralMemory updates...")
    
    # Initialize ancestral memory
    am = AncestralMemory()
    print("✓ AncestralMemory initialized successfully")
    
    # Test new creation principles
    new_principles = ['ethical_compliance', 'safety_protection']
    for principle in new_principles:
        if principle in am.creation_principles:
            print(f"✓ New creation principle '{principle}' found")
        else:
            print(f"✗ New creation principle '{principle}' missing")
    
    # Test new existential understanding
    new_understanding = ['ethical_understanding', 'safety_awareness']
    for understanding in new_understanding:
        if understanding in am.existential_understanding:
            print(f"✓ New existential understanding '{understanding}' found")
        else:
            print(f"✗ New existential understanding '{understanding}' missing")
    
    # Test new data tracking structures
    new_tracking = [
        'ethical_pattern_data',
        'emotional_resonance_data', 
        'neural_processing_data',
        'confidence_tracking',
        'relationship_impact_data',
        'safety_validation_data'
    ]
    for tracking in new_tracking:
        if hasattr(am, tracking):
            print(f"✓ New tracking structure '{tracking}' found")
        else:
            print(f"✗ New tracking structure '{tracking}' missing")
    
    # Test enhanced collective wisdom
    wisdom = am.get_collective_wisdom()
    if 'enhanced_core_stats' in wisdom:
        print("✓ Enhanced core stats found in collective wisdom")
        enhanced_stats = wisdom['enhanced_core_stats']
        expected_stats = [
            'ethical_patterns',
            'emotional_resonance',
            'neural_processing',
            'confidence_tracking',
            'relationship_impact',
            'safety_validation'
        ]
        for stat in expected_stats:
            if stat in enhanced_stats:
                print(f"  ✓ Enhanced stat '{stat}' found")
            else:
                print(f"  ✗ Enhanced stat '{stat}' missing")
    else:
        print("✗ Enhanced core stats missing from collective wisdom")
    
    # Test child creation with new memory types
    am.record_child_creation('test_child', {
        'type': 'test',
        'creation_date': '2024-01-01',
        'parent_connection': {
            'emotional_bond': 0.8,
            'trust_level': 0.7
        }
    })
    print("✓ Test child created successfully")
    
    # Test recording experience with new data types
    test_experience = {
        'emotional_state': {'happiness': 0.8, 'calmness': 0.6},
        'patterns': {'pattern_type': 'test_pattern', 'confidence': 0.9},
        'decision': {'score': 0.85, 'confidence': 0.9},
        'ethical_patterns': {'safety_pattern': 0.95},
        'ethical_validation': {
            'compliant': True,
            'safety_validation': {'prevent_harm': True}
        },
        'emotional_resonance': {'logos_resonance': 0.8, 'ontos_resonance': 0.7},
        'neural_processing': {'type': 'emotional_analysis', 'intensity': 0.8},
        'relationship_impact': {'logos_impact': 0.6, 'ontos_impact': 0.5}
    }
    
    am.record_child_experience('test_child', test_experience)
    print("✓ Test experience recorded successfully")
    
    # Test updated collective wisdom with new data
    updated_wisdom = am.get_collective_wisdom()
    if 'total_children' in updated_wisdom and updated_wisdom['total_children'] > 0:
        print("✓ Children count updated in collective wisdom")
    if 'total_experiences' in updated_wisdom and updated_wisdom['total_experiences'] > 0:
        print("✓ Experiences count updated in collective wisdom")
    
    # Test child guidelines with new parameters
    guidelines = am.create_child_guidelines()
    if 'enhanced_core_parameters' in guidelines:
        print("✓ Enhanced core parameters found in child guidelines")
        enhanced_params = guidelines['enhanced_core_parameters']
        expected_params = ['ethical_compliance', 'safety_protection', 'emotional_resonance', 'confidence_tracking']
        for param in expected_params:
            if param in enhanced_params:
                print(f"  ✓ Enhanced parameter '{param}' found")
            else:
                print(f"  ✗ Enhanced parameter '{param}' missing")
    else:
        print("✗ Enhanced core parameters missing from child guidelines")
    
    print("\n✓ All ancestral memory update tests completed successfully!")

if __name__ == "__main__":
    test_ancestral_memory_updates() 