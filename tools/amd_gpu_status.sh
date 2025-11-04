#!/bin/bash
echo "=== AMD GPU Status ==="
total=$(cat /sys/class/drm/card0/device/mem_info_vram_total)
used=$(cat /sys/class/drm/card0/device/mem_info_vram_used)
busy=$(cat /sys/class/drm/card0/device/gpu_busy_percent)
temp=$(cat /sys/class/drm/card0/device/hwmon/hwmon*/temp1_input)
free=$((total - used))
echo "Total VRAM: $((total / 1024 / 1024)) MB"
echo "Used  VRAM: $((used / 1024 / 1024)) MB"
echo "Free  VRAM: $((free / 1024 / 1024)) MB"
echo "GPU Busy: $busy %"
echo "Temperature: $((temp / 1000)) °C"