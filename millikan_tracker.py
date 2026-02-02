#!/usr/bin/env python3
"""
Millikan Oil Drop Experiment - Automated Droplet Tracker
Tracks oil droplets and calculates fall/rise velocities
"""

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from collections import defaultdict
import json
import os

class MillikanDropletTracker:
    def __init__(self, video_path, major_line_spacing_mm=0.5):
        """
        Initialize the tracker

        Args:
            video_path: Path to the video file
            major_line_spacing_mm: Spacing between major reticle lines in mm
        """
        self.video_path = video_path
        self.major_line_spacing_mm = major_line_spacing_mm
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calibration (pixels per mm)
        self.pixels_per_mm = None
        self.roi = None  # Region of interest (the grid area)

        # Tracking data
        self.tracks = defaultdict(list)  # track_id -> [(frame, x, y, time), ...]
        self.next_track_id = 0

        print(f"Video: {os.path.basename(video_path)}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Duration: {self.frame_count/self.fps:.2f}s ({self.frame_count} frames)")

    def calibrate_from_reticle(self, frame):
        """
        Detect the reticle grid and calibrate pixels per mm
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the circular viewfield (bright region)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            # Focus on center region where grid is clearest
            margin = 50
            self.roi = (x + margin, y + margin, x + w - margin, y + h - margin)
        else:
            # Default ROI
            margin = 100
            self.roi = (margin, margin, self.width - margin, self.height - margin)

        # Extract ROI for grid detection
        x1, y1, x2, y2 = self.roi
        roi_gray = gray[y1:y2, x1:x2]

        # Detect grid lines using edge detection
        edges = cv2.Canny(roi_gray, 30, 100)

        # Use Hough transform to find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=50, maxLineGap=10)

        if lines is not None:
            # Separate horizontal and vertical lines
            h_lines = []
            v_lines = []

            for line in lines:
                x1_l, y1_l, x2_l, y2_l = line[0]
                angle = np.abs(np.arctan2(y2_l - y1_l, x2_l - x1_l) * 180 / np.pi)

                if angle < 20 or angle > 160:  # Horizontal
                    h_lines.append((y1_l + y2_l) / 2)
                elif 70 < angle < 110:  # Vertical
                    v_lines.append((x1_l + x2_l) / 2)

            # Find spacing between lines
            if len(h_lines) > 3:
                h_lines = sorted(set([int(y) for y in h_lines]))
                # Find major line spacing (look for consistent larger gaps)
                h_diffs = np.diff(h_lines)

                # The major lines should have larger spacing
                # Minor lines are typically at 0.1mm, major at 0.5mm (5x minor)
                median_spacing = np.median(h_diffs)

                # Group similar spacings
                minor_spacings = h_diffs[h_diffs < median_spacing * 1.5]
                if len(minor_spacings) > 0:
                    minor_spacing_px = np.median(minor_spacings)
                    # 5 minor divisions = 1 major division = 0.5mm
                    major_spacing_px = minor_spacing_px * 5
                    self.pixels_per_mm = major_spacing_px / self.major_line_spacing_mm
                else:
                    # Use median as estimate
                    self.pixels_per_mm = median_spacing / 0.1  # Assume minor spacing

        # Fallback calibration if detection failed
        if self.pixels_per_mm is None or self.pixels_per_mm < 10:
            # Estimate based on typical microscope setup
            # Typical: grid spans ~3mm in view, view is ~400px
            roi_height = y2 - y1
            estimated_field_mm = 3.0  # Estimate visible field is ~3mm
            self.pixels_per_mm = roi_height / estimated_field_mm
            print(f"  Using estimated calibration: {self.pixels_per_mm:.1f} px/mm")
        else:
            print(f"  Calibrated: {self.pixels_per_mm:.1f} pixels/mm")

        return self.pixels_per_mm

    def detect_droplets(self, frame, prev_frame=None):
        """
        Detect oil droplets in a frame
        Returns list of (x, y, radius, brightness) for each detected droplet
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply ROI mask
        if self.roi:
            x1, y1, x2, y2 = self.roi
            mask = np.zeros_like(gray)
            mask[y1:y2, x1:x2] = 255
            gray = cv2.bitwise_and(gray, mask)

        # Background subtraction if we have previous frame
        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            if self.roi:
                prev_gray = cv2.bitwise_and(prev_gray, mask)

            # Compute difference to highlight moving objects
            diff = cv2.absdiff(gray, prev_gray)
            # Combine with current frame (droplets are bright spots)
            enhanced = cv2.addWeighted(gray, 0.5, diff, 2.0, 0)
        else:
            enhanced = gray.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Droplets appear as bright spots - use adaptive threshold
        # Calculate local mean to find spots brighter than surroundings
        local_mean = cv2.blur(blurred, (51, 51))
        diff_from_mean = cv2.subtract(blurred, local_mean)

        # Threshold to find bright spots
        _, binary = cv2.threshold(diff_from_mean, 8, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        droplets = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            # Filter by size (droplets are small but not tiny noise)
            if 3 < area < 500:
                cx, cy = centroids[i]
                # Estimate radius from area
                radius = np.sqrt(area / np.pi)
                # Get brightness at centroid
                brightness = gray[int(cy), int(cx)] if 0 <= int(cy) < gray.shape[0] and 0 <= int(cx) < gray.shape[1] else 0
                droplets.append((cx, cy, radius, brightness))

        return droplets

    def track_droplets(self, max_frames=None, progress_interval=100):
        """
        Track droplets throughout the video
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, first_frame = self.cap.read()
        if not ret:
            print("Error reading video")
            return

        # Calibrate using first frame
        self.calibrate_from_reticle(first_frame)

        # Reset video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        prev_frame = None
        prev_droplets = []
        frame_num = 0

        if max_frames is None:
            max_frames = self.frame_count

        print(f"  Tracking droplets...")

        while frame_num < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect droplets in current frame
            current_droplets = self.detect_droplets(frame, prev_frame)
            time_sec = frame_num / self.fps

            # Match droplets to existing tracks
            if prev_droplets and current_droplets:
                # Simple nearest-neighbor matching
                matched_current = set()

                for track_id, prev_positions in list(self.tracks.items()):
                    if not prev_positions:
                        continue

                    last_frame, last_x, last_y, _ = prev_positions[-1]

                    # Only match if last seen recently
                    if frame_num - last_frame > 15:  # Allow 0.5s gap at 30fps
                        continue

                    # Predict position (assume roughly constant velocity)
                    if len(prev_positions) >= 2:
                        dx = prev_positions[-1][1] - prev_positions[-2][1]
                        dy = prev_positions[-1][2] - prev_positions[-2][2]
                        frames_gap = frame_num - last_frame
                        pred_x = last_x + dx * frames_gap
                        pred_y = last_y + dy * frames_gap
                    else:
                        pred_x, pred_y = last_x, last_y

                    # Find nearest droplet to prediction
                    best_dist = float('inf')
                    best_idx = -1

                    for idx, (cx, cy, r, b) in enumerate(current_droplets):
                        if idx in matched_current:
                            continue
                        dist = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
                        # Allow reasonable movement (droplets move slowly)
                        max_dist = 50 * (frame_num - last_frame)  # pixels
                        if dist < best_dist and dist < max_dist:
                            best_dist = dist
                            best_idx = idx

                    if best_idx >= 0:
                        cx, cy, _, _ = current_droplets[best_idx]
                        self.tracks[track_id].append((frame_num, cx, cy, time_sec))
                        matched_current.add(best_idx)

                # Start new tracks for unmatched droplets
                for idx, (cx, cy, r, b) in enumerate(current_droplets):
                    if idx not in matched_current:
                        self.tracks[self.next_track_id].append((frame_num, cx, cy, time_sec))
                        self.next_track_id += 1

            elif current_droplets:
                # First frame with droplets - start tracks
                for cx, cy, r, b in current_droplets:
                    self.tracks[self.next_track_id].append((frame_num, cx, cy, time_sec))
                    self.next_track_id += 1

            prev_frame = frame.copy()
            prev_droplets = current_droplets
            frame_num += 1

            if frame_num % progress_interval == 0:
                print(f"    Processed {frame_num}/{max_frames} frames...")

        self.cap.release()
        print(f"  Tracking complete. Found {len(self.tracks)} potential tracks.")

        return self.tracks

    def analyze_velocities(self, min_track_length=10):
        """
        Analyze track data to extract fall and rise velocities

        In Millikan's experiment:
        - Fall = droplet moving DOWN (gravity, no field or field pushing down)
        - Rise = droplet moving UP (electric field pulling up)

        Returns DataFrame with velocity data for each droplet
        """
        results = []

        for track_id, positions in self.tracks.items():
            if len(positions) < min_track_length:
                continue

            # Convert to numpy array
            data = np.array(positions)
            frames = data[:, 0]
            x_pos = data[:, 1]
            y_pos = data[:, 2]
            times = data[:, 3]

            # Calculate velocities (pixels per frame)
            dy = np.diff(y_pos)
            dt = np.diff(times)

            # Avoid division by zero
            valid = dt > 0
            if not np.any(valid):
                continue

            vy_pixels = dy[valid] / dt[valid]  # pixels per second

            # Convert to mm/s
            if self.pixels_per_mm:
                vy_mm = vy_pixels / self.pixels_per_mm
            else:
                vy_mm = vy_pixels / 100  # Fallback estimate

            # Segment into rising and falling periods
            # Positive y = downward in image coordinates
            # Rising: dy < 0 (moving up in image)
            # Falling: dy > 0 (moving down in image)

            # Find segments of consistent motion
            segments = []
            current_segment = {'type': None, 'velocities': [], 'start_time': None, 'end_time': None}

            for i, vy in enumerate(vy_mm):
                if abs(vy) < 0.001:  # Nearly stationary
                    if current_segment['velocities']:
                        segments.append(current_segment.copy())
                        current_segment = {'type': None, 'velocities': [], 'start_time': None, 'end_time': None}
                    continue

                motion_type = 'fall' if vy > 0 else 'rise'

                if current_segment['type'] is None:
                    current_segment['type'] = motion_type
                    current_segment['start_time'] = times[i]

                if motion_type == current_segment['type']:
                    current_segment['velocities'].append(abs(vy))
                    current_segment['end_time'] = times[i + 1]
                else:
                    if current_segment['velocities']:
                        segments.append(current_segment.copy())
                    current_segment = {
                        'type': motion_type,
                        'velocities': [abs(vy)],
                        'start_time': times[i],
                        'end_time': times[i + 1]
                    }

            if current_segment['velocities']:
                segments.append(current_segment)

            # Extract fall and rise velocities
            fall_velocities = []
            rise_velocities = []

            for seg in segments:
                if len(seg['velocities']) >= 3:  # Need enough points for reliable velocity
                    avg_vel = np.median(seg['velocities'])  # Use median for robustness
                    duration = seg['end_time'] - seg['start_time'] if seg['start_time'] and seg['end_time'] else 0

                    if seg['type'] == 'fall':
                        fall_velocities.append({
                            'velocity_mm_s': avg_vel,
                            'duration_s': duration,
                            'start_time': seg['start_time'],
                            'n_points': len(seg['velocities'])
                        })
                    else:
                        rise_velocities.append({
                            'velocity_mm_s': avg_vel,
                            'duration_s': duration,
                            'start_time': seg['start_time'],
                            'n_points': len(seg['velocities'])
                        })

            if fall_velocities or rise_velocities:
                results.append({
                    'track_id': track_id,
                    'total_frames': len(positions),
                    'duration_s': times[-1] - times[0],
                    'fall_velocities': fall_velocities,
                    'rise_velocities': rise_velocities,
                    'n_fall_segments': len(fall_velocities),
                    'n_rise_segments': len(rise_velocities),
                    'avg_fall_velocity_mm_s': np.mean([f['velocity_mm_s'] for f in fall_velocities]) if fall_velocities else None,
                    'avg_rise_velocity_mm_s': np.mean([r['velocity_mm_s'] for r in rise_velocities]) if rise_velocities else None
                })

        return results


def process_video(video_path, output_dir):
    """Process a single video and return results"""
    tracker = MillikanDropletTracker(video_path)
    tracker.track_droplets()
    results = tracker.analyze_velocities(min_track_length=5)

    # Filter to significant tracks
    significant = [r for r in results if r['duration_s'] > 0.5]

    return {
        'video': os.path.basename(video_path),
        'fps': tracker.fps,
        'pixels_per_mm': tracker.pixels_per_mm,
        'total_tracks': len(tracker.tracks),
        'significant_tracks': len(significant),
        'droplet_data': significant
    }


def main():
    """Process all trial videos"""
    video_files = [
        '/sessions/inspiring-dreamy-ride/mnt/uploads/trial1drop1.mov',
        '/sessions/inspiring-dreamy-ride/mnt/uploads/trial2.mov',
        '/sessions/inspiring-dreamy-ride/mnt/uploads/trial3.mov',
        '/sessions/inspiring-dreamy-ride/mnt/uploads/trial4.mov'
    ]

    output_dir = '/sessions/inspiring-dreamy-ride/mnt/Oil_Drop'
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for video_path in video_files:
        if os.path.exists(video_path):
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(video_path)}")
            print('='*60)

            try:
                result = process_video(video_path, output_dir)
                all_results.append(result)

                # Print summary for this video
                print(f"\n  Results Summary:")
                print(f"    Significant droplet tracks: {result['significant_tracks']}")

                for drop in result['droplet_data']:
                    print(f"\n    Droplet (Track {drop['track_id']}):")
                    print(f"      Duration: {drop['duration_s']:.2f}s, Frames: {drop['total_frames']}")
                    if drop['avg_fall_velocity_mm_s']:
                        print(f"      Avg Fall Velocity: {drop['avg_fall_velocity_mm_s']:.4f} mm/s ({drop['n_fall_segments']} segments)")
                    if drop['avg_rise_velocity_mm_s']:
                        print(f"      Avg Rise Velocity: {drop['avg_rise_velocity_mm_s']:.4f} mm/s ({drop['n_rise_segments']} segments)")

            except Exception as e:
                print(f"  Error processing {video_path}: {e}")
                import traceback
                traceback.print_exc()

    # Save detailed results to JSON
    json_path = os.path.join(output_dir, 'millikan_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nDetailed results saved to: {json_path}")

    # Create summary CSV
    csv_data = []
    for result in all_results:
        video_name = result['video']
        for drop in result['droplet_data']:
            # Add individual velocity measurements
            for i, fall in enumerate(drop['fall_velocities']):
                csv_data.append({
                    'video': video_name,
                    'track_id': drop['track_id'],
                    'motion_type': 'fall',
                    'segment': i + 1,
                    'velocity_mm_s': fall['velocity_mm_s'],
                    'duration_s': fall['duration_s'],
                    'start_time_s': fall['start_time'],
                    'n_points': fall['n_points']
                })
            for i, rise in enumerate(drop['rise_velocities']):
                csv_data.append({
                    'video': video_name,
                    'track_id': drop['track_id'],
                    'motion_type': 'rise',
                    'segment': i + 1,
                    'velocity_mm_s': rise['velocity_mm_s'],
                    'duration_s': rise['duration_s'],
                    'start_time_s': rise['start_time'],
                    'n_points': rise['n_points']
                })

    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, 'millikan_velocities.csv')
        df.to_csv(csv_path, index=False)
        print(f"Velocity data saved to: {csv_path}")

        # Print summary table
        print("\n" + "="*80)
        print("SUMMARY OF ALL VELOCITY MEASUREMENTS")
        print("="*80)
        print(df.to_string(index=False))

    return all_results


if __name__ == '__main__':
    results = main()
