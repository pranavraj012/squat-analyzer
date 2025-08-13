import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Dict, Any

class ProcessFrameFront:
    def __init__(self, thresholds: Dict[str, Any] = None, flip_frame: bool = False, skill_level: str = 'beginner', strict: bool = False):
        # Config/context
        self.flip_frame = flip_frame
        self.thresholds = thresholds or {}
        self.skill_level = skill_level.lower()
        self.strict = strict

        # State tracking
        self.state = 'S1'
        self.prev_state = 'S1'
        self.rep_count = 0
        self.incorrect_count = 0
        self.depth_reached = False
        self.feedback_msgs = []

        # Persistent feedback (message -> frames left)
        self.message_persistence = {}
        self.message_duration_frames = 50

        # Stability / smoothing
        self.state_counter = 0
        self.min_state_frames = 3
        self.current_state_frames = 0
        self.angle_history = []
        self.max_history = 5

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Landmark indices
        self.LEFT_SHOULDER = 11; self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23; self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25; self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27; self.RIGHT_ANKLE = 28
        self.NOSE = 0

        # Relaxed thresholds (strict slightly tighter)
        self.front_thresholds = {
            'knee_valgus': { 'beginner': 20.0, 'professional': 15.0, 'severe': 30.0 },
            'hip_asym': { 'beginner': 35, 'professional': 25 },
            'shoulder_asym': { 'beginner': 40, 'professional': 30 },
            'lateral_lean': { 'beginner': 55, 'professional': 40 },
            'stance_narrow_ratio': { 'beginner': 0.7, 'professional': 0.85 },
            'stance_wide_ratio': { 'beginner': 2.0, 'professional': 1.8 },
        }
        if self.strict:
            self.front_thresholds['knee_valgus']['beginner'] -= 3
            self.front_thresholds['knee_valgus']['professional'] -= 2
            self.front_thresholds['hip_asym']['beginner'] -= 5
            self.front_thresholds['hip_asym']['professional'] -= 5
            self.front_thresholds['lateral_lean']['beginner'] -= 5
            self.front_thresholds['lateral_lean']['professional'] -= 5

        if 'FRONT_VIEW' in self.thresholds:
            for k, v in self.thresholds['FRONT_VIEW'].items():
                if k in self.front_thresholds and isinstance(v, dict):
                    self.front_thresholds[k].update(v)

        # Consecutive frame gating
        self.issue_counters = {}
        self.frames_required = 6

    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate the angle (in degrees) at point b given three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def get_landmark_coords(self, landmarks, idx: int, shape: Tuple[int, int]) -> Tuple[int, int]:
        """Convert normalized landmark to pixel coordinates."""
        h, w = shape
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)

    def _add_message(self, msg: str):
        """Add or refresh a feedback message with persistence."""
        self.message_persistence[msg] = self.message_duration_frames

    def _decay_messages(self):
        remove = []
        for k in self.message_persistence:
            self.message_persistence[k] -= 1
            if self.message_persistence[k] <= 0:
                remove.append(k)
        for k in remove:
            del self.message_persistence[k]

    def process_landmarks(self, landmarks, shape: Tuple[int, int]) -> Dict[str, Any]:
        """Process landmarks and analyze squat form (front view)."""
        # Get coordinates
        l_hip = self.get_landmark_coords(landmarks, self.LEFT_HIP, shape)
        r_hip = self.get_landmark_coords(landmarks, self.RIGHT_HIP, shape)
        l_knee = self.get_landmark_coords(landmarks, self.LEFT_KNEE, shape)
        r_knee = self.get_landmark_coords(landmarks, self.RIGHT_KNEE, shape)
        l_ankle = self.get_landmark_coords(landmarks, self.LEFT_ANKLE, shape)
        r_ankle = self.get_landmark_coords(landmarks, self.RIGHT_ANKLE, shape)
        l_shoulder = self.get_landmark_coords(landmarks, self.LEFT_SHOULDER, shape)
        r_shoulder = self.get_landmark_coords(landmarks, self.RIGHT_SHOULDER, shape)
        nose = self.get_landmark_coords(landmarks, self.NOSE, shape)
        
        # Calculate knee angles
        l_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
        r_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_angle + r_angle) / 2
        
        # Smooth angle using history to reduce noise
        self.angle_history.append(avg_knee_angle)
        if len(self.angle_history) > self.max_history:
            self.angle_history.pop(0)
        
        # Use smoothed angle for more stable detection
        smoothed_angle = sum(self.angle_history) / len(self.angle_history)
        
        # State machine for squat detection with improved logic
        prev_state = self.state
        
        # Determine new state based on smoothed angle with hysteresis
        new_state = self.state
        if smoothed_angle > 160:  # Lowered from 165 for better detection
            new_state = 'S1'  # Standing
        elif smoothed_angle < 100:  # Raised from 95 for clearer deep squat
            new_state = 'S3'  # Deep squat
        else:
            new_state = 'S2'  # Mid-squat
        
        # Simplified state change logic - less rigid
        if new_state != self.state:
            self.current_state_frames += 1
            # Only need 2-3 consistent frames for state change
            if self.current_state_frames >= self.min_state_frames:
                # DON'T update prev_state here - do it after rep counting
                self.state = new_state
                self.current_state_frames = 0
        else:
            self.current_state_frames = 0
        
        # Improved rep counting logic with proper form validation
        form_issues = []
        skill = 'professional' if self.skill_level.startswith('pro') else 'beginner'
        ft = self.front_thresholds
        
        # Track form issues that affect rep quality (only during squat motion)
        rep_affecting_issues = []

        # Helper to gate issues by consecutive frames & phases
        def gate_issue(condition: bool, key: str, message: str, phase_restricted: bool = True, affects_rep: bool = False):
            # Only evaluate during squat descent/bottom unless stance-related
            if phase_restricted and self.state not in ['S2', 'S3']:
                self.issue_counters[key] = 0
                return
            if condition:
                self.issue_counters[key] = self.issue_counters.get(key, 0) + 1
                if self.issue_counters[key] == self.frames_required:
                    self._add_message(message)
                    if affects_rep:
                        rep_affecting_issues.append(key)
            else:
                self.issue_counters[key] = 0

        # ---- Dynamic Knee Valgus (FPPA) ----
        fppa_left = 180.0 - l_angle
        fppa_right = 180.0 - r_angle
        severe_valgus = ft['knee_valgus']['severe']
        valgus_thresh = ft['knee_valgus'][skill]
        valgus_flags = {'left': False, 'right': False, 'severe': False}
        if fppa_left > valgus_thresh: valgus_flags['left'] = True
        if fppa_right > valgus_thresh: valgus_flags['right'] = True
        if fppa_left > severe_valgus or fppa_right > severe_valgus: valgus_flags['severe'] = True
        gate_issue(valgus_flags['left'] or valgus_flags['right'], 'valgus', 'PUSH KNEES OUT', affects_rep=True)
        if valgus_flags['severe']:
            self._add_message('STOP - KNEES IN')
            rep_affecting_issues.append('valgus')

        # ---- Hip Asymmetry ----
        hip_diff = abs(l_hip[1] - r_hip[1])
        gate_issue(hip_diff > ft['hip_asym'][skill], 'hip_asym', 'LEVEL YOUR HIPS', affects_rep=True)

        # ---- Shoulder Alignment (make optional & less intrusive) ----
        shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
        if self.strict:
            gate_issue(shoulder_diff > ft['shoulder_asym'][skill], 'shoulder_asym', 'LEVEL SHOULDERS')
        else:
            shoulder_diff = 0  # hide in analysis flags when relaxed

        # ---- Stance Width (only check in standing state early) ----
        hip_width = abs(l_hip[0] - r_hip[0]) + 1e-6
        foot_width = abs(l_ankle[0] - r_ankle[0])
        stance_ratio = foot_width / hip_width
        if self.state == 'S1':
            gate_issue(stance_ratio < ft['stance_narrow_ratio'][skill], 'stance', 'ADJUST STANCE', phase_restricted=False)
            gate_issue(stance_ratio > ft['stance_wide_ratio'][skill], 'stance', 'ADJUST STANCE', phase_restricted=False)
        else:
            self.issue_counters['stance'] = 0

        # ---- Lateral Upper Body Lean ----
        pelvis_center = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)
        lateral_disp = abs(nose[0] - pelvis_center[0])
        gate_issue(lateral_disp > ft['lateral_lean'][skill], 'lateral_lean', 'CENTER TORSO', affects_rep=True)
        
        # Check for knee valgus (knees caving in) during squat
        # (Legacy simple checks retained if needed)
    # Legacy simple check removed (redundant) to reduce noise
        
        # Mark depth reached with proper conditions
        if self.state == 'S3' and smoothed_angle < 110:
            self.depth_reached = True
        
        # Add feedback for insufficient depth during squat attempt
        if self.state == 'S2' and not self.depth_reached:
            # User is in mid-squat but hasn't reached deep position yet
            gate_issue(True, 'shallow', 'GO DEEPER', phase_restricted=False)
        
        # Rep counting logic with form validation - counts shallow squats as incorrect
        if (self.prev_state in ['S2', 'S3'] and self.state == 'S1' and smoothed_angle > 155):
            # Only count if we actually attempted a squat (went to S2 or deeper)
            if self.depth_reached and len(rep_affecting_issues) == 0:
                # Good form rep - reached proper depth with good form
                self.rep_count += 1
            elif self.depth_reached and len(rep_affecting_issues) > 0:
                # Bad form rep - reached depth but had form issues
                self.incorrect_count += 1
            elif not self.depth_reached and self.prev_state in ['S2', 'S3']:
                # Shallow squat - attempted but didn't reach proper depth, count as incorrect
                self.incorrect_count += 1
            # Reset depth flag for next rep
            self.depth_reached = False
        
        # Update prev_state AFTER rep counting to prevent multiple counts for same rep
        self.prev_state = self.state
        
        # Decay persistent messages and collect current feedback list (sorted for stability)
        self._decay_messages()
        self.feedback_msgs = sorted(self.message_persistence.keys())
        
        return {
            'state': self.state,
            'rep_count': self.rep_count,
            'feedback': self.feedback_msgs,
            'landmarks': {
                'l_hip': l_hip, 'r_hip': r_hip,
                'l_knee': l_knee, 'r_knee': r_knee,
                'l_ankle': l_ankle, 'r_ankle': r_ankle
            },
            'angles': {
                'left_knee': l_angle,
                'right_knee': r_angle,
                'avg_knee': avg_knee_angle,
                'fppa_left': fppa_left,
                'fppa_right': fppa_right
            },
            'flags': {
                'valgus': valgus_flags,
                'hip_diff': hip_diff,
                'shoulder_diff': shoulder_diff,
                'stance_ratio': stance_ratio,
                'lateral_disp': lateral_disp
            }
        }

    def draw_overlay(self, frame: np.ndarray, results, analysis: Dict[str, Any]) -> np.ndarray:
        """Draw pose landmarks and analysis overlay on frame (front view)."""
        frame_height, frame_width = frame.shape[:2]
        
        # Draw pose landmarks
        if results.pose_landmarks:
            # MediaPipe drawing utilities expect RGB format, so colors are specified in RGB
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

            # Extract landmarks for custom overlays
            lm = results.pose_landmarks.landmark
            h, w = frame_height, frame_width
            def pt(i): return int(lm[i].x * w), int(lm[i].y * h)
            l_knee = pt(self.LEFT_KNEE); r_knee = pt(self.RIGHT_KNEE)
            l_ankle = pt(self.LEFT_ANKLE); r_ankle = pt(self.RIGHT_ANKLE)
            l_hip = pt(self.LEFT_HIP); r_hip = pt(self.RIGHT_HIP)
            nose = pt(self.NOSE)
            pelvis_center = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)

            # Center alignment line
            cv2.line(frame, (pelvis_center[0], 0), (pelvis_center[0], frame_height), (255, 255, 0), 1)
            # Vertical ankle reference lines
            cv2.line(frame, (l_ankle[0], 0), (l_ankle[0], frame_height), (0, 180, 255), 1)
            cv2.line(frame, (r_ankle[0], 0), (r_ankle[0], frame_height), (0, 180, 255), 1)

            # Knee tracking indicators (green good, red valgus)
            valgus_flags = analysis.get('flags', {}).get('valgus', {})
            for knee_pt, side in [(l_knee, 'left'), (r_knee, 'right')]:
                color = (0, 255, 0)
                if valgus_flags.get(side):
                    color = (255, 0, 0) if not valgus_flags.get('severe') else (255, 0, 255)
                cv2.circle(frame, knee_pt, 10, color, 2)

            # Lateral lean line (nose to pelvis center)
            cv2.line(frame, nose, pelvis_center, (0, 200, 255), 2)

            # Real-time FPPA display
            angles = analysis.get('angles', {})
            fppa_left = angles.get('fppa_left', 0.0)
            fppa_right = angles.get('fppa_right', 0.0)
            cv2.putText(frame, f"L-FPPA:{int(fppa_left)}", (l_knee[0]-40, l_knee[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,255,255), 2)
            cv2.putText(frame, f"R-FPPA:{int(fppa_right)}", (r_knee[0]-40, r_knee[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,255,255), 2)
        
        # Display CORRECT and INCORRECT counts like ProcessFrame (top right corner)
        # Using the same positioning and styling as ProcessFrame
        correct_text = f"CORRECT: {self.rep_count}"
        incorrect_text = f"INCORRECT: {self.incorrect_count}"
        
        # Create background rectangles for text (matching ProcessFrame style)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text sizes for background rectangles
        (correct_w, correct_h), _ = cv2.getTextSize(correct_text, font, font_scale, thickness)
        (incorrect_w, incorrect_h), _ = cv2.getTextSize(incorrect_text, font, font_scale, thickness)
        
        # Position at top right (matching ProcessFrame positioning)
        correct_x = int(frame_width * 0.68)
        correct_y = 30
        incorrect_x = int(frame_width * 0.68)
        incorrect_y = 80
        
        # Draw background rectangles
        cv2.rectangle(frame, (correct_x - 5, correct_y - correct_h - 5), 
                     (correct_x + correct_w + 5, correct_y + 5), (0, 185, 18), -1)
        cv2.rectangle(frame, (incorrect_x - 5, incorrect_y - incorrect_h - 5), 
                     (incorrect_x + incorrect_w + 5, incorrect_y + 5), (0, 0, 221), -1)
        
        # Draw text (RGB colors for RGB frame)
        cv2.putText(frame, correct_text, (correct_x, correct_y), 
                   font, font_scale, (230, 255, 255), thickness)
        cv2.putText(frame, incorrect_text, (incorrect_x, incorrect_y), 
                   font, font_scale, (230, 255, 255), thickness)
        
        # Display feedback messages at the bottom left (persistent & colored by severity)
        y = frame_height - 60
        for msg in analysis['feedback']:
            severity_color = (0, 153, 255)  # default orange
            if 'SEVERE' in msg or 'STOP' in msg:
                severity_color = (255, 0, 80)  # red/pink severe
            elif 'KNEES' in msg or 'TORSO' in msg:
                severity_color = (255, 80, 80)  # red-ish
            (msg_w, msg_h), _ = cv2.getTextSize(msg, font, 0.55, 2)
            cv2.rectangle(frame, (25, y - msg_h - 5), (35 + msg_w, y + 5), severity_color, -1)
            cv2.putText(frame, msg, (30, y), font, 0.55, (230, 255, 255), 2)
            y -= 30
        
        return frame

    def process(self, frame: np.ndarray, pose_model) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single frame and return annotated frame with analysis."""
        if self.flip_frame:
            frame = cv2.flip(frame, 1)
        
        # Expect RGB input consistently (matching ProcessFrame behavior)
        # This aligns with the upload processing logic
        frame_rgb = frame.copy()
        
        results = pose_model.process(frame_rgb)
        
        # Use RGB frame for processing and display
        output_frame = frame_rgb.copy()
        
        analysis = {'state': self.state, 'rep_count': self.rep_count, 'feedback': []}
        
        if results.pose_landmarks:
            analysis = self.process_landmarks(results.pose_landmarks.landmark, output_frame.shape[:2])
            output_frame = self.draw_overlay(output_frame, results, analysis)
        
        # Return RGB output consistently (matching ProcessFrame behavior)
        return output_frame, analysis

    def reset(self):
        """Reset the analyzer state."""
        self.state = 'S1'
        self.prev_state = 'S1'
        self.rep_count = 0
        self.incorrect_count = 0  # Reset incorrect count too
        self.depth_reached = False
        self.feedback_msgs = []
        self.state_counter = 0
        self.current_state_frames = 0
        self.angle_history = []
        self.message_persistence = {}
        self.issue_counters = {}