#!/usr/bin/env python3
"""Check available L515 stream configurations"""
import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("No RealSense devices found!")
    exit(1)

dev = devices[0]
print(f"Device: {dev.get_info(rs.camera_info.name)}")
print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
print(f"Firmware: {dev.get_info(rs.camera_info.firmware_version)}")

for sensor in dev.query_sensors():
    print(f"\nSensor: {sensor.get_info(rs.camera_info.name)}")
    for profile in sensor.get_stream_profiles():
        if profile.stream_type() in [rs.stream.color, rs.stream.depth]:
            vp = profile.as_video_stream_profile()
            print(f"  {profile.stream_type()}: {vp.width()}x{vp.height()} @ {vp.fps()}fps, {profile.format()}")
