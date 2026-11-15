#!/usr/bin/env python3
"""
run.py
======
NASA International Space Apps Challenge — Global Competition Submission
Interplanetary Survival Guide: Martian Map

Single-Command Launcher for the Complete Mission Operations System:
- Dual HTTP/HTTPS Web Server (Ports 8080 / 8443)
- Full-Duplex WebSocket Telemetry Bridge (Ports 8081 / 8444)
- 3D Mission Control Cockpit (frontend/public/index.html)
- Mobile Field EVA Companion & Gyro HUD (frontend/public/phone.html)
- NASA Rover Street-View & Photo Proximity Engine (mars_locator.py)

Usage:
  python run.py
"""

import os
import sys
import socket
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
APP_DIR = ROOT_DIR / "martian-map"


def get_lan_ip() -> str:
    """Detects active local LAN IP for seamless mobile pairing."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def print_banner(lan_ip: str):
    print("=" * 76)
    print("   _  _   _   ___   _     __  __   _   ___ _____ ___   _   _  _ ")
    print("  | \\| | /_\\ / __| /_\\   |  \\/  | /_\\ | _ \\_   _|_ _| /_\\ | \\| |")
    print("  | .` |/ _ \\\\__ \\/ _ \\  | |\\/| |/ _ \\|   / | |  | | / _ \\| .` |")
    print("  |_|\\_/_/ \\_\\___/_/ \\_\\ |_|  |_/_/ \\_\\_|_\\ |_| |___/_/ \\_\\_|\\_|")
    print("      INTERPLANETARY SURVIVAL GUIDE: AUTONOMOUS MARTIAN MAP")
    print("       NASA International Space Apps Challenge — Global Finalist")
    print("=" * 76)
    print("\n[MISSION READY] System Initialized Successfully:")
    print(f"  • Mission Control 3D Cockpit : http://localhost:8080")
    print(f"  • Secure Cockpit (SSL)       : https://localhost:8443")
    print(f"  • Mobile Field EVA Gyro HUD  : http://{lan_ip}:8080/phone.html")
    print(f"  • Mobile Secure (SSL)        : https://{lan_ip}:8443/phone.html")
    print(f"  • WebSocket Telemetry Stream : ws://localhost:8081  (Secure: wss://8444)")
    print("\n[OPERATIONAL TIPS]:")
    print("  1. Desktop: Hover terrain or click TARGET / REFILL to plan bioenergetic paths.")
    print("  2. Mobile: Scan the 'Pair Gyro' QR code on desktop to steer 3D view with phone IMU.")
    print("  3. Press Ctrl+C in this terminal to shut down all mission systems.")
    print("=" * 76 + "\n")


def check_preflight():
    bundle_path = APP_DIR / "frontend" / "public" / "data" / "cesium_mars_mission_bundle.json"
    if not bundle_path.exists():
        print(f"[WARNING] Precomputed mission bundle not found at {bundle_path}.")
        print("  Generating mission bundle via pipeline/run_full_pipeline.py...")
        import subprocess
        res = subprocess.run([sys.executable, "pipeline/run_full_pipeline.py"], cwd=str(APP_DIR))
        if res.returncode != 0:
            print("[ERROR] Failed to generate bundle. Please verify pipeline configuration.")
            sys.exit(1)


def main():
    check_preflight()
    lan_ip = get_lan_ip()
    print_banner(lan_ip)

    # Change working directory to martian-map so relative paths align
    os.chdir(str(APP_DIR))
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))

    # Execute server
    import server
    server.main()


if __name__ == "__main__":
    main()
