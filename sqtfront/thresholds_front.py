"""
Thresholds configuration for front view squat analysis.
These thresholds are used to determine proper form and provide feedback.
"""

def get_thresholds_beginner_front():
    """Get thresholds for beginner mode front view analysis."""
    return {
        'knee_angle_standing': 165,  # Minimum angle for standing position
        'knee_angle_deep': 95,       # Maximum angle for deep squat position
        'knee_valgus_threshold': 0.95,  # Ratio for detecting knee cave-in
        'hip_asymmetry_threshold': 20,  # Maximum pixel difference for hip alignment
        'depth_required': True,      # Whether deep position is required for rep count
        'feedback_enabled': True,    # Whether to show form feedback
        'angle_display': True,       # Whether to display knee angles
    }

def get_thresholds_pro_front():
    """Get thresholds for professional mode front view analysis."""
    return {
        'knee_angle_standing': 170,  # Stricter standing position
        'knee_angle_deep': 90,       # Deeper squat requirement
        'knee_valgus_threshold': 0.98,  # Stricter knee alignment
        'hip_asymmetry_threshold': 15,  # Stricter hip alignment
        'depth_required': True,      # Deep position required
        'feedback_enabled': True,    # Show detailed feedback
        'angle_display': True,       # Show angle measurements
        'form_strictness': 'high',   # Higher form standards
    }