import cv2
import numpy as np
import pygame
import csv
from datetime import datetime
import os

# ============================================================================
# PHASE 1: SETUP & AUDIO INITIALIZATION
# ============================================================================

def initialize_audio():
    """Initialize pygame mixer and load sound effects"""
    pygame.mixer.init()
    
    hit_path = os.path.abspath('hit.wav')
    miss_path = os.path.abspath('miss.wav')
    
    hit_sound = None
    miss_sound = None
    
    if os.path.exists(hit_path):
        try:
            hit_sound = pygame.mixer.Sound(hit_path)
            print(f"✓ Hit sound loaded: {hit_path}")
        except Exception as e:
            print(f"⚠ Error loading hit sound: {e}")
    else:
        print(f"⚠ Warning: hit.wav not found at {hit_path}")
    
    if os.path.exists(miss_path):
        try:
            miss_sound = pygame.mixer.Sound(miss_path)
            print(f"✓ Miss sound loaded: {miss_path}")
        except Exception as e:
            print(f"⚠ Error loading miss sound: {e}")
    else:
        print(f"⚠ Warning: miss.wav not found at {miss_path}")
    
    if hit_sound is None and miss_sound is None:
        print("⚠ No audio files found. Running in silent mode.")
    
    return hit_sound, miss_sound

def initialize_webcam():
    """Initialize webcam capture"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Could not open webcam")
    print("✓ Webcam initialized")
    return cap

# ============================================================================
# PHASE 2: TARGET CALIBRATION
# ============================================================================

class TargetCalibrator:
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.dragging = False
        self.calibrated = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse drag for boundary selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.dragging = True
            
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.dragging = False
            print(f"Target area selected: {self.start_point} to {self.end_point}")
            
    def calibrate(self, frame):
        """Manual calibration: User drags a rectangle around target"""
        display_frame = frame.copy()
        
        cv2.putText(display_frame, "Drag to select target area, then press SPACE", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if self.start_point and self.end_point:
            cv2.rectangle(display_frame, self.start_point, self.end_point, (0, 255, 0), 2)
            
            width = abs(self.end_point[0] - self.start_point[0])
            height = abs(self.end_point[1] - self.start_point[1])
            mid_x = (self.start_point[0] + self.end_point[0]) // 2
            mid_y = (self.start_point[1] + self.end_point[1]) // 2
            cv2.putText(display_frame, f"{width}x{height}px", 
                       (mid_x - 40, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return display_frame
    
    def get_perspective_transform(self):
        """Create perspective transformation matrix"""
        if not self.start_point or not self.end_point:
            return None
        
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        top_left = (min(x1, x2), min(y1, y2))
        bottom_right = (max(x1, x2), max(y1, y2))
        
        src_pts = np.float32([
            top_left,
            (bottom_right[0], top_left[1]),
            bottom_right,
            (top_left[0], bottom_right[1])
        ])
        
        dst_pts = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.calibrated = True
        print("✓ Calibration complete - Ready to shoot!")
        return matrix

# ============================================================================
# PHASE 3: SCORING GEOMETRY
# ============================================================================

class ScoringSystem:
    def __init__(self):
        self.center = (250, 250)
        self.scoring_rings = [
            (20, 10),   # Bullseye
            (50, 8),
            (100, 6),
            (150, 4),
            (200, 2),
        ]
        
    def calculate_score(self, hit_point):
        """Calculate score based on distance from center"""
        if hit_point is None:
            return 0, 999
            
        dx = hit_point[0] - self.center[0]
        dy = hit_point[1] - self.center[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Check if within scoring rings
        for radius, points in self.scoring_rings:
            if distance <= radius:
                return points, distance
        
        # Outside rings but inside 500x500 = 0 points
        return 0, distance
    
    def is_hit(self, hit_point):
        """Check if point is within scoring rings (distance < 200)"""
        if hit_point is None:
            return False
        
        dx = hit_point[0] - self.center[0]
        dy = hit_point[1] - self.center[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        return distance < 200
    
    def draw_target_rings(self, frame):
        """Draw scoring rings on frame"""
        colors = [(255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255)]
        
        for i, (radius, points) in enumerate(reversed(self.scoring_rings)):
            color = colors[i % len(colors)]
            cv2.circle(frame, self.center, radius, color, 2)
            
        cv2.circle(frame, self.center, 3, (0, 0, 255), -1)
        return frame

# ============================================================================
# PHASE 4: PULSE DETECTION - FIRST APPEARANCE RULE
# ============================================================================

class PulseDetector:
    def __init__(self):
        self.is_laser_present = False
        self.brightness_threshold = 220  # Aggressive threshold
        
    def detect_laser(self, normalized_view):
        """Detect laser using aggressive brightness threshold
        Only detects within the 500x500 normalized view"""
        
        # Convert to HSV and extract Value channel
        hsv = cv2.cvtColor(normalized_view, cv2.COLOR_BGR2HSV)
        _, _, v_channel = cv2.split(hsv)
        
        # Aggressive threshold: Target bright spots regardless of color
        _, bright_mask = cv2.threshold(v_channel, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        
        return bright_mask
    
    def get_first_appearance_coordinate(self, bright_mask):
        """Get the (x, y) coordinate of the laser ONLY on first appearance
        State Machine:
        - is_laser_present = False + detection = NEW SHOT (capture coordinate)
        - is_laser_present = True + detection = SAME SHOT (do nothing)
        - is_laser_present = True + no detection = READY FOR NEXT SHOT
        """
        
        # Find contours
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        laser_detected = len(contours) > 0
        shot_coordinate = None
        
        # STATE MACHINE LOGIC
        if laser_detected:
            # Laser is currently visible
            if not self.is_laser_present:
                # NEW SHOT - First appearance!
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shot_coordinate = (cx, cy)
                    self.is_laser_present = True
                    print(f"🔴 LASER FIRST APPEARANCE at ({cx}, {cy})")
            # else: is_laser_present = True, so this is the same shot being held
        else:
            # No laser detected
            if self.is_laser_present:
                # Laser just disappeared - ready for next shot
                self.is_laser_present = False
                print("✓ Laser removed - Ready for next shot")
        
        return shot_coordinate, laser_detected

# ============================================================================
# PHASE 5: 3-SHOT ROUND MANAGER
# ============================================================================

class ThreeShotRound:
    def __init__(self, hit_sound, miss_sound, scoring_system):
        self.hit_sound = hit_sound
        self.miss_sound = miss_sound
        self.scoring_system = scoring_system
        
        self.shot_count = 0
        self.total_score = 0
        self.shot_markers = []  # Store (x, y, score) tuples
        self.round_finished = False
        
    def process_new_shot(self, coordinate, normalized_view):
        """Process a new shot when first appearance is detected"""
        
        # Check if round is already finished
        if self.shot_count >= 3:
            return
        
        # Increment shot count
        self.shot_count += 1
        
        # Calculate score
        score, distance = self.scoring_system.calculate_score(coordinate)
        self.total_score += score
        
        # Determine if hit or miss
        is_hit = self.scoring_system.is_hit(coordinate)
        
        # Play sound
        if is_hit:
            if self.hit_sound:
                self.hit_sound.play()
            status = "HIT"
            print(f"🎯 Shot {self.shot_count}: HIT! Score: {score} (Distance: {distance:.1f}px)")
        else:
            if self.miss_sound:
                self.miss_sound.play()
            status = "MISS"
            print(f"❌ Shot {self.shot_count}: MISS (Distance: {distance:.1f}px)")
        
        # Store marker
        self.shot_markers.append((coordinate[0], coordinate[1], score))
        
        # Save image
        self.save_shot_image(normalized_view, coordinate, self.shot_count, score)
        
        # Log to CSV
        self.log_shot(coordinate, score, distance, status)
        
        # Check if round finished
        if self.shot_count >= 3:
            self.round_finished = True
            print(f"\n🏁 ROUND FINISHED! Total Score: {self.total_score}/30")
        
        return {
            'shot': self.shot_count,
            'coordinate': coordinate,
            'score': score,
            'distance': distance,
            'status': status
        }
    
    def save_shot_image(self, frame, coordinate, shot_number, score):
        """Save image with permanent red circle at hit coordinate"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shot_{shot_number}_{timestamp}.jpg"
        
        screenshot = frame.copy()
        
        # Draw permanent red circle at hit coordinate
        cv2.circle(screenshot, coordinate, 12, (0, 0, 255), 3)
        cv2.circle(screenshot, coordinate, 3, (0, 0, 255), -1)
        
        # Add shot number and score
        cv2.putText(screenshot, f"Shot {shot_number}: {score}pts", 
                   (coordinate[0] + 15, coordinate[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imwrite(filename, screenshot)
        print(f"  📸 Saved: {filename}")
    
    def log_shot(self, coordinate, score, distance, status):
        """Append shot data to log.csv immediately"""
        filename = 'log.csv'
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Shot', 'X', 'Y', 'Score', 'Distance', 'Status'])
        
        # Append shot data
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.shot_count,
                coordinate[0],
                coordinate[1],
                score,
                f"{distance:.2f}",
                status
            ])
    
    def reset_round(self):
        """Reset for a new 3-shot round"""
        print("\n🔄 Starting new round...")
        self.shot_count = 0
        self.total_score = 0
        self.shot_markers = []
        self.round_finished = False
    
    def draw_ui(self, frame):
        """Draw UI elements on frame"""
        
        # Background for HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Shot counter
        shots_remaining = 3 - self.shot_count
        cv2.putText(frame, f"Shots Remaining: {shots_remaining}/3", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current score
        cv2.putText(frame, f"Score: {self.total_score}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw all shot markers
        for i, (x, y, score) in enumerate(self.shot_markers):
            color = (0, 0, 255)  # Red
            cv2.circle(frame, (x, y), 10, color, 2)
            cv2.circle(frame, (x, y), 2, color, -1)
            cv2.putText(frame, str(i+1), (x + 12, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Round finished message
        if self.round_finished:
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (50, 180), (450, 320), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)
            
            cv2.putText(frame, "ROUND FINISHED!", 
                       (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Total Score: {self.total_score}/30", 
                       (120, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'N' for new round", 
                       (110, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    print("=" * 60)
    print("SHOOTING RANGE - 3 SHOT ROUNDS")
    print("First-Frame Detection System")
    print("=" * 60)
    
    # Initialize
    cap = initialize_webcam()
    hit_sound, miss_sound = initialize_audio()
    
    # Calibration
    calibrator = TargetCalibrator()
    cv2.namedWindow('Shooting Range')
    cv2.setMouseCallback('Shooting Range', calibrator.mouse_callback)
    
    # Systems
    scoring = ScoringSystem()
    detector = PulseDetector()
    round_manager = ThreeShotRound(hit_sound, miss_sound, scoring)
    
    transform_matrix = None
    
    print("\n📋 Instructions:")
    print("1. Drag to select target area, press SPACE")
    print("2. Point laser at target - first appearance = shot!")
    print("3. After 3 shots, press 'N' for new round")
    print("4. Press 'Q' to quit")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # CALIBRATION MODE
        if not calibrator.calibrated:
            display = calibrator.calibrate(frame)
            cv2.imshow('Shooting Range', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and calibrator.start_point and calibrator.end_point:
                transform_matrix = calibrator.get_perspective_transform()
            elif key == ord('q'):
                break
        
        # SHOOTING MODE
        else:
            # Get 500x500 normalized view (detection only happens here)
            normalized_view = cv2.warpPerspective(frame, transform_matrix, (500, 500))
            
            # Draw target rings
            display_frame = scoring.draw_target_rings(normalized_view.copy())
            
            # Detect laser (aggressive threshold, target area only)
            bright_mask = detector.detect_laser(normalized_view)
            
            # Get first appearance coordinate (state machine)
            shot_coordinate, laser_active = detector.get_first_appearance_coordinate(bright_mask)
            
            # Process new shot if coordinate captured
            if shot_coordinate is not None and round_manager.shot_count < 3:
                round_manager.process_new_shot(shot_coordinate, normalized_view)
            
            # Draw UI
            display_frame = round_manager.draw_ui(display_frame)
            
            # Show laser indicator if active
            if laser_active:
                cv2.putText(display_frame, "LASER ACTIVE", (350, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Display
            cv2.imshow('Shooting Range', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                round_manager.reset_round()
                detector.is_laser_present = False
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    
    print("\n" + "=" * 60)
    print("Session ended. Check log.csv for shot data.")
    print("=" * 60)

if __name__ == "__main__":
    main()