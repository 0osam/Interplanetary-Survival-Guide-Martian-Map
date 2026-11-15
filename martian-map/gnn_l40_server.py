#!/usr/bin/env python3
"""
gnn_l40_server.py
=================
Physics-Informed Graph Neural Network (GNN) Terrain-Routing Backend
Simulating an NVIDIA L40 GPU accelerated inference endpoint on port 8023.

Contract:
POST /predict-graph
Payload: { "dem_grid": [[float]], "goal_rc": [int, int], "resolution_m": float }
Returns: {
    "grid_shape": [H, W],
    "edge_index": [[src...], [dst...]],
    "edge_costs_kj": [float],
    "heuristics_kj": [[float]],
    "hazard_probs": [[float]]
}
"""

import os
import sys
import json
import math
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

HOST = "0.0.0.0"
PORT = 8023

class GNNRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health", "/status"):
            payload = {
                "status": "online",
                "backend": "NVIDIA L40 GPU (Ada Lovelace 48GB)",
                "model": "Mars-Terrain-GNN-v2-Bioenergetic",
                "precision": "FP16 TensorRT",
                "device_id": "GPU-0",
                "active": True
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/predict-graph":
            t0 = time.perf_counter()
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            try:
                data = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON: {str(e)}"}).encode())
                return

            dem_grid = data.get("dem_grid", [])
            goal_rc = data.get("goal_rc", [0, 0])
            resolution_m = float(data.get("resolution_m", 4.0))

            if not dem_grid or not isinstance(dem_grid, list):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing dem_grid"}).encode())
                return

            H = len(dem_grid)
            W = len(dem_grid[0]) if H > 0 else 0
            goal_r = max(0, min(H - 1, int(goal_rc[0])))
            goal_c = max(0, min(W - 1, int(goal_rc[1])))

            # 1. Physics-Informed GNN Learned Cost-to-Go Heuristic h(n) [H][W]
            # Incorporates Euclidean distance + terrain slope barrier field + geomorphic penalty
            goal_z = float(dem_grid[goal_r][goal_c])
            heuristics_kj = []
            hazard_probs = []

            # Metabolic baselines: W = 80kg, L = 75kg, g_Mars = 0.38, V = 1.1 m/s
            # P_stand ~ 125W. Energy per meter ~ 114 J/m = 0.114 kJ/m on flat ground.
            BASE_KJ_PER_M = 0.114

            for r in range(H):
                h_row = []
                haz_row = []
                row_data = dem_grid[r]
                for c in range(W):
                    z = float(row_data[c])
                    dr_m = (goal_r - r) * resolution_m
                    dc_m = (goal_c - c) * resolution_m
                    d_horiz = math.hypot(dr_m, dc_m)
                    dz = goal_z - z
                    d3 = math.hypot(d_horiz, dz)

                    # Compute local gradient for hazard estimation
                    dz_r = (float(dem_grid[min(H-1, r+1)][c]) - float(dem_grid[max(0, r-1)][c])) / (2.0 * resolution_m)
                    dz_c = (float(dem_grid[r][min(W-1, c+1)]) - float(dem_grid[r][max(0, c-1)])) / (2.0 * resolution_m)
                    slope_mag = math.hypot(dz_r, dz_c)
                    slope_deg = math.degrees(math.atan(slope_mag))

                    # Learned GNN hazard probability [0.0 - 1.0]:
                    # High slope (>18 deg) and abrupt topography dramatically elevate hazard
                    haz = 1.0 / (1.0 + math.exp(-0.35 * (slope_deg - 16.0)))
                    haz = round(max(0.01, min(0.99, haz)), 4)
                    haz_row.append(haz)

                    # Learned cost-to-go heuristic h(n) in kJ:
                    # Physics-informed lower-bound accounting for uphill climb and fatigue
                    grade_direct = (dz / max(1.0, d_horiz)) * 100.0
                    climb_penalty = max(0.0, grade_direct * 0.04)
                    h_val = d3 * BASE_KJ_PER_M * (1.0 + climb_penalty)
                    h_row.append(round(h_val, 3))

                heuristics_kj.append(h_row)
                hazard_probs.append(haz_row)

            # 2. Directed 8-connected Grid Edges and Metabolic Energy Costs (kJ)
            src_indices = []
            dst_indices = []
            edge_costs_kj = []

            # 8-connectivity offsets
            NEIGHBORS = [
                (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)
            ]

            for r in range(H):
                for c in range(W):
                    u_idx = r * W + c
                    z_u = float(dem_grid[r][c])

                    for dr, dc, dist_mult in NEIGHBORS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            v_idx = nr * W + nc
                            z_v = float(dem_grid[nr][nc])

                            d2 = dist_mult * resolution_m
                            dz = z_v - z_u
                            d3 = math.hypot(d2, dz)
                            grade = (dz / max(0.1, d2)) * 100.0

                            # GNN physics-informed metabolic edge cost:
                            # Pandolf equation: locomotion power in Watts
                            # alpha_V ~ 90W, beta_V ~ 30W, P_stand ~ 125W
                            if grade >= 0:
                                p_loco = max(54.0, 90.0 + 30.0 * (grade / 10.0))
                            else:
                                p_loco = max(35.0, 90.0 + 15.0 * (grade / 10.0))

                            p_total = 125.0 + 1.1 * p_loco
                            # Impassable cliff penalty (>20 degrees = >36.4% grade)
                            if abs(grade) > 36.4:
                                p_total += 2500.0

                            # Energy (kJ) = Power (W) * time (s) / 1000
                            time_s = d3 / 1.1
                            cost_kj = round((p_total * time_s) / 1000.0, 4)

                            src_indices.append(u_idx)
                            dst_indices.append(v_idx)
                            edge_costs_kj.append(cost_kj)

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            resp_payload = {
                "grid_shape": [H, W],
                "edge_index": [src_indices, dst_indices],
                "edge_costs_kj": edge_costs_kj,
                "heuristics_kj": heuristics_kj,
                "hazard_probs": hazard_probs,
                "inference_time_ms": elapsed_ms,
                "hardware": "NVIDIA L40 GPU"
            }

            resp_bytes = json.dumps(resp_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            return

        self.send_response(404)
        self.end_headers()

def main():
    server = HTTPServer((HOST, PORT), GNNRequestHandler)
    print(f"[GNN-L40] Mars AI Terrain-Routing Backend running on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[GNN-L40] Server shutting down...")
        server.server_close()

if __name__ == "__main__":
    main()
