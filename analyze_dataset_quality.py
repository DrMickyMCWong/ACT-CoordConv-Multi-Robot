#!/usr/bin/env python3
"""
Dataset Quality Analysis for Reactive Obstacle Avoidance
Analyzes QP filter residuals to identify episodes with good obstacle interactions
"""

import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

# Load HDF5 function (copied from notebook)
def load_hdf5(dataset_path):
    """Load robotics episode data from HDF5 file with support for multiple action types."""
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        return None, None, None, None, None, None
    
    with h5py.File(dataset_path, 'r') as root:
        # Load joint data
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        
        # Load camera images
        image_dict = dict()
        if 'observations/images' in root:
            for cam_name in root[f'/observations/images/'].keys():
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        
        # Load all action types for QP filter analysis
        action_data = {}
        action_data['action'] = root['/action'][()]  # Final QP-filtered actions
        
        # Load additional action types if available
        if '/action_raw' in root:
            action_data['action_raw'] = root['/action_raw'][()]
        if '/action_processed' in root:
            action_data['action_processed'] = root['/action_processed'][()]
        if '/action_filtered' in root:
            action_data['action_filtered'] = root['/action_filtered'][()]
            
        # Load point cloud data if available
        pointcloud_dict = {}
        if 'observations/pointclouds' in root:
            for cam_name in root['observations/pointclouds'].keys():
                pointcloud_dict[cam_name] = {
                    'points_camera': root[f'observations/pointclouds/{cam_name}/points_camera'][()],
                    'points_world': root[f'observations/pointclouds/{cam_name}/points_world'][()], 
                    'num_points': root[f'observations/pointclouds/{cam_name}/num_points'][()]
                }
            
        # Load metadata attributes
        attrs = {}
        for key in root.attrs.keys():
            attrs[key] = root.attrs[key]
    
    return qpos, qvel, action_data, image_dict, attrs, pointcloud_dict

def analyze_episode_quality(episode_path):
    """Analyze QP filter activity and obstacle interaction quality."""
    print(f"Analyzing: {os.path.basename(episode_path)}")
    
    try:
        qpos, qvel, action_data, image_dict, attrs, pointcloud_dict = load_hdf5(episode_path)
        
        if qpos is None or 'action_processed' not in action_data:
            return None
            
        # Calculate residuals (QP filter effect)
        action_processed = action_data['action_processed']  # Before QP filter
        action_final = action_data['action']  # After QP filter
        residuals = action_final - action_processed
        
        # Focus on arm joints (first 6 DOF) - QP filter doesn't affect gripper
        arm_residuals = residuals[:, :6]
        
        # Calculate quality metrics
        mean_residual = np.mean(np.abs(arm_residuals))
        max_residual = np.max(np.abs(arm_residuals))
        std_residual = np.std(np.abs(arm_residuals))
        
        # Count timesteps with significant QP activity (threshold: 0.01 rad)
        active_timesteps = np.sum(np.any(np.abs(arm_residuals) > 0.01, axis=1))
        total_timesteps = len(residuals)
        
        # Quality score: percentage of timesteps with QP activity
        quality_score = active_timesteps / total_timesteps
        
        # Obstacle interaction phases
        high_activity = np.sum(np.any(np.abs(arm_residuals) > 0.05, axis=1))  # Strong QP corrections
        medium_activity = np.sum(np.any(np.abs(arm_residuals) > 0.02, axis=1))  # Moderate corrections
        
        return {
            'episode': os.path.basename(episode_path),
            'mean_residual': mean_residual,
            'max_residual': max_residual,
            'std_residual': std_residual,
            'active_timesteps': active_timesteps,
            'total_timesteps': total_timesteps,
            'quality_score': quality_score,
            'high_activity': high_activity,
            'medium_activity': medium_activity,
            'obstacle_height': attrs.get('obstacle_height', 0.0),
            'task_name': attrs.get('task_name', 'unknown'),
            'success': attrs.get('success', False)
        }
        
    except Exception as e:
        print(f"Error analyzing {episode_path}: {e}")
        return None

def analyze_dataset_directory(dataset_dir):
    """Analyze all episodes in a directory."""
    episode_files = glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))
    episode_files.sort()
    
    print(f"📊 Analyzing {len(episode_files)} episodes in {dataset_dir}")
    print("=" * 80)
    
    results = []
    for episode_file in episode_files:
        result = analyze_episode_quality(episode_file)
        if result:
            results.append(result)
    
    return results

def create_quality_report(results):
    """Generate comprehensive quality report."""
    if not results:
        print("No valid episodes found!")
        return
    
    # Convert to arrays for analysis
    quality_scores = np.array([r['quality_score'] for r in results])
    mean_residuals = np.array([r['mean_residual'] for r in results])
    obstacle_heights = np.array([r['obstacle_height'] for r in results])
    
    print("\n" + "="*80)
    print("📈 DATASET QUALITY REPORT")
    print("="*80)
    
    # Overall statistics
    print(f"Total episodes: {len(results)}")
    print(f"Mean quality score: {np.mean(quality_scores):.3f} ± {np.std(quality_scores):.3f}")
    print(f"Mean residual magnitude: {np.mean(mean_residuals):.4f} ± {np.std(mean_residuals):.4f}")
    
    # Quality categorization
    high_quality = np.sum(quality_scores > 0.3)  # >30% QP activity
    medium_quality = np.sum((quality_scores > 0.1) & (quality_scores <= 0.3))  # 10-30%
    low_quality = np.sum(quality_scores <= 0.1)  # <10%
    
    print(f"\nQuality Distribution:")
    print(f"  🟢 High quality (>30% QP activity): {high_quality}/{len(results)} ({high_quality/len(results)*100:.1f}%)")
    print(f"  🟡 Medium quality (10-30% QP activity): {medium_quality}/{len(results)} ({medium_quality/len(results)*100:.1f}%)")
    print(f"  🔴 Low quality (<10% QP activity): {low_quality}/{len(results)} ({low_quality/len(results)*100:.1f}%)")
    
    # Height-based analysis
    unique_heights = np.unique(obstacle_heights)
    print(f"\nObstacle Height Analysis:")
    for height in sorted(unique_heights):
        height_mask = obstacle_heights == height
        height_episodes = np.sum(height_mask)
        height_quality = np.mean(quality_scores[height_mask])
        print(f"  Height {height:.3f}m: {height_episodes} episodes, avg quality: {height_quality:.3f}")
    
    # Top and bottom episodes
    sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
    
    print(f"\n🏆 TOP 5 EPISODES (Best obstacle interactions):")
    for i in range(min(5, len(results))):
        idx = sorted_indices[i]
        r = results[idx]
        print(f"  {r['episode']}: Quality={r['quality_score']:.3f}, Height={r['obstacle_height']:.3f}m, "
              f"Active={r['active_timesteps']}/{r['total_timesteps']}")
    
    print(f"\n🔴 BOTTOM 5 EPISODES (Poorest obstacle interactions):")
    for i in range(min(5, len(results))):
        idx = sorted_indices[-(i+1)]
        r = results[idx]
        print(f"  {r['episode']}: Quality={r['quality_score']:.3f}, Height={r['obstacle_height']:.3f}m, "
              f"Active={r['active_timesteps']}/{r['total_timesteps']}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if np.mean(quality_scores) < 0.25:
        print("  ⚠️  Dataset has low obstacle interaction density")
        print("  💡 Collect more episodes at heights 0.08-0.12m for better interactions")
    
    optimal_heights = []
    for height in unique_heights:
        height_mask = obstacle_heights == height
        if np.sum(height_mask) > 0:
            avg_quality = np.mean(quality_scores[height_mask])
            if avg_quality > 0.3:
                optimal_heights.append(height)
    
    if optimal_heights:
        print(f"  ✅ Best heights for collection: {optimal_heights}")
    else:
        print("  💡 Try intermediate heights (0.08m, 0.12m) for better QP interactions")
    
    # CRITICAL ANALYSIS: Free-space behavior
    print(f"\n🚨 REACTIVE POLICY CONCERN ANALYSIS:")
    free_space_episodes = np.sum(quality_scores < 0.05)  # <5% QP activity = mostly free-space
    print(f"  📊 Free-space dominated episodes: {free_space_episodes}/{len(results)} ({free_space_episodes/len(results)*100:.1f}%)")
    
    if free_space_episodes > 0:
        # Analyze free-space behavior
        free_mask = quality_scores < 0.05
        free_results = [results[i] for i in range(len(results)) if free_mask[i]]
        
        print(f"  🔍 Free-space episode analysis:")
        for r in free_results[:3]:  # Show first 3 examples
            print(f"    - {r['episode']}: {r['active_timesteps']}/{r['total_timesteps']} active steps")
        
        print(f"\n  💡 SOLUTION STRATEGY:")
        print(f"    1. ✅ Include {free_space_episodes} free-space episodes in training")
        print(f"    2. ✅ Ensure reactive policy learns: 'no obstacle → no correction'")
        print(f"    3. ✅ Test on obstacle-free environments to verify pass-through behavior")
        print(f"    4. ⚠️  If policy adds unnecessary corrections in free space, reduce training weight on low-residual episodes")
    
    balance_ratio = (len(results) - free_space_episodes) / len(results)
    print(f"\n  📈 DATASET BALANCE:")
    print(f"    - Obstacle interaction episodes: {balance_ratio*100:.1f}%")
    print(f"    - Free-space episodes: {(1-balance_ratio)*100:.1f}%")
    
    if balance_ratio < 0.3:
        print(f"    🔴 RISK: Too few obstacle interactions - policy may not learn reactive behavior")
    elif balance_ratio > 0.8:
        print(f"    🟡 CAUTION: Too few free-space examples - policy may over-correct in obstacle-free scenarios")
    else:
        print(f"    🟢 GOOD: Balanced dataset for reactive policy training")
    
    return results

def plot_quality_analysis(results):
    """Create visualizations of dataset quality."""
    if not results:
        return
    
    # Extract data
    quality_scores = [r['quality_score'] for r in results]
    mean_residuals = [r['mean_residual'] for r in results]
    obstacle_heights = [r['obstacle_height'] for r in results]
    episodes = [r['episode'] for r in results]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Quality score distribution
    axes[0, 0].hist(quality_scores, bins=20, alpha=0.7, color='skyblue')
    axes[0, 0].axvline(np.mean(quality_scores), color='red', linestyle='--', label=f'Mean: {np.mean(quality_scores):.3f}')
    axes[0, 0].set_xlabel('Quality Score (% QP Activity)')
    axes[0, 0].set_ylabel('Number of Episodes')
    axes[0, 0].set_title('Dataset Quality Distribution')
    axes[0, 0].legend()
    
    # Plot 2: Quality vs Obstacle Height
    scatter = axes[0, 1].scatter(obstacle_heights, quality_scores, c=mean_residuals, cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel('Obstacle Height (m)')
    axes[0, 1].set_ylabel('Quality Score')
    axes[0, 1].set_title('Quality vs Obstacle Height')
    plt.colorbar(scatter, ax=axes[0, 1], label='Mean Residual')
    
    # Plot 3: Episode-wise quality
    episode_indices = range(len(episodes))
    bars = axes[1, 0].bar(episode_indices, quality_scores, alpha=0.7)
    
    # Color bars by quality
    for i, bar in enumerate(bars):
        if quality_scores[i] > 0.3:
            bar.set_color('green')
        elif quality_scores[i] > 0.1:
            bar.set_color('orange') 
        else:
            bar.set_color('red')
    
    axes[1, 0].set_xlabel('Episode Index')
    axes[1, 0].set_ylabel('Quality Score')
    axes[1, 0].set_title('Episode Quality Overview')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Residual magnitude distribution
    axes[1, 1].hist(mean_residuals, bins=20, alpha=0.7, color='lightcoral')
    axes[1, 1].axvline(np.mean(mean_residuals), color='darkred', linestyle='--', label=f'Mean: {np.mean(mean_residuals):.4f}')
    axes[1, 1].set_xlabel('Mean Residual Magnitude (rad)')
    axes[1, 1].set_ylabel('Number of Episodes')
    axes[1, 1].set_title('QP Filter Activity Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main analysis function."""
    # Default dataset directory
    dataset_dir = '/home/hk/Documents/ACT_Shaka/act-main/act/ckpt_sim_pick_cube_teleop/eval_data'
    
    # Allow command line override
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    # Analyze dataset
    results = analyze_dataset_directory(dataset_dir)
    
    # Generate report
    create_quality_report(results)
    
    # Create visualizations
    plot_quality_analysis(results)
    
    # Save results for further analysis
    import pickle
    results_file = os.path.join(dataset_dir, 'quality_analysis.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()