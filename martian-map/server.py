import os
import sys
import json
import time
import socket
import select
import struct
import base64
import hashlib
import threading
import ssl
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import urllib.request
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from mars_locator import find_nearby_images

WS_MAGIC = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
HTTP_PORT = 8080
HTTPS_PORT = 8443
WS_PORT = 8081
WSS_PORT = 8444

PUBLIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "frontend", "public"))
CERT_FILE = os.path.join(BASE_DIR, "cert.pem")
KEY_FILE = os.path.join(BASE_DIR, "key.pem")

# Thread-safe in-memory state
state_lock = threading.Lock()
latest_phone_telemetry = {
    "type": "phone_input",
    "heading": 0.0,
    "alpha": 0.0,
    "beta": 0.0,
    "gamma": 0.0,
    "walk": 0,
    "speed": 1.0,
    "command": None,
    "source": "init",
    "seq": 0,
    "updated_at": time.time()
}
latest_pc_telemetry = {
    "type": "pc_telemetry",
    "currO2": 25.0,
    "tankCapacityL": 25.0,
    "elevLocal": 0.0,
    "carriedSampleKg": 0.0,
    "pnrStatus": "SAFE (1.15×)",
    "pathProgress": 0.0,
    "pathDist": 0.0,
    "isPathWalking": False,
    "heading": 0.0,
    "updated_at": time.time()
}

ws_clients = set()
ws_lock = threading.Lock()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def encode_ws_frame(data, opcode=1):
    b = data.encode("utf-8") if isinstance(data, str) else data
    length = len(b)
    if length <= 125:
        hdr = bytes([0x80 | opcode, length])
    elif length <= 65535:
        hdr = bytes([0x80 | opcode, 126]) + struct.pack("!H", length)
    else:
        hdr = bytes([0x80 | opcode, 127]) + struct.pack("!Q", length)
    return hdr + b

def decode_ws_frame(buf):
    if len(buf) < 2:
        return None, 0
    b1, b2 = buf[0], buf[1]
    opcode = b1 & 0x0F
    masked = (b2 >> 7) & 1
    plen = b2 & 0x7F
    offset = 2
    if plen == 126:
        if len(buf) < 4:
            return None, 0
        plen = struct.unpack("!H", buf[2:4])[0]
        offset = 4
    elif plen == 127:
        if len(buf) < 10:
            return None, 0
        plen = struct.unpack("!Q", buf[2:10])[0]
        offset = 10
    if masked:
        if len(buf) < offset + 4:
            return None, 0
        mask = buf[offset:offset+4]
        offset += 4
    if len(buf) < offset + plen:
        return None, 0
    payload = buf[offset:offset+plen]
    if masked:
        payload = bytes([b ^ mask[i % 4] for i, b in enumerate(payload)])
    return (opcode, payload), offset + plen

def broadcast_ws(payload_str, exclude_sock=None):
    frame = encode_ws_frame(payload_str, opcode=1)
    with ws_lock:
        dead = set()
        for client in ws_clients:
            if client != exclude_sock:
                try:
                    client.sendall(frame)
                except Exception:
                    dead.add(client)
        ws_clients.difference_update(dead)

def ws_client_worker(sock, addr):
    buf = bytearray()
    handshake_done = False
    try:
        sock.settimeout(60.0)
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                try:
                    sock.sendall(encode_ws_frame(b"ping", opcode=9))
                except Exception:
                    break
                continue
            if not chunk:
                break
            buf.extend(chunk)

            if not handshake_done:
                if b"\r\n\r\n" in buf:
                    headers_part, _, remaining = bytes(buf).partition(b"\r\n\r\n")
                    buf = bytearray(remaining)
                    headers_text = headers_part.decode("latin-1", errors="ignore")
                    key = None
                    for line in headers_text.splitlines():
                        if line.lower().startswith("sec-websocket-key:"):
                            key = line.split(":", 1)[1].strip()
                            break
                    if key:
                        acc = base64.b64encode(hashlib.sha1(key.encode() + WS_MAGIC).digest()).decode()
                        resp = (
                            "HTTP/1.1 101 Switching Protocols\r\n"
                            "Upgrade: websocket\r\n"
                            "Connection: Upgrade\r\n"
                            f"Sec-WebSocket-Accept: {acc}\r\n\r\n"
                        ).encode()
                        sock.sendall(resp)
                        handshake_done = True
                        with ws_lock:
                            ws_clients.add(sock)
                        print(f"[WS] Client connected: {addr} (Total: {len(ws_clients)})")
                        with state_lock:
                            init_payload = json.dumps({
                                "type": "sync_state",
                                "phone": latest_phone_telemetry,
                                "pc": latest_pc_telemetry,
                                **latest_phone_telemetry
                            })
                        sock.sendall(encode_ws_frame(init_payload))
                    else:
                        break

            if handshake_done:
                while True:
                    frame, consumed = decode_ws_frame(buf)
                    if frame is None:
                        break
                    buf = buf[consumed:]
                    opcode, payload = frame
                    if opcode == 8: # close
                        return
                    elif opcode == 9: # ping -> send pong
                        sock.sendall(encode_ws_frame(payload, opcode=10))
                    elif opcode == 1: # text
                        msg = payload.decode("utf-8", errors="ignore")
                        try:
                            data = json.loads(msg)
                            msg_type = data.get("type")
                            with state_lock:
                                if msg_type == "phone_input":
                                    latest_phone_telemetry.update(data)
                                    latest_phone_telemetry["updated_at"] = time.time()
                                elif msg_type == "pc_telemetry":
                                    latest_pc_telemetry.update(data)
                                    latest_pc_telemetry["updated_at"] = time.time()
                                else:
                                    latest_phone_telemetry.update(data)
                                    latest_phone_telemetry["updated_at"] = time.time()
                        except Exception:
                            pass
                        broadcast_ws(msg, exclude_sock=sock)
    except Exception:
        pass
    finally:
        with ws_lock:
            ws_clients.discard(sock)
        try:
            sock.close()
        except Exception:
            pass
        print(f"[WS] Client disconnected: {addr} (Remaining: {len(ws_clients)})")

def run_ws_server(port, ssl_ctx=None):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_sock.bind(("0.0.0.0", port))
        server_sock.listen(16)
        proto = "wss" if ssl_ctx else "ws"
        print(f"[{proto.upper()}] Telemetry Relay running on {proto}://0.0.0.0:{port}")
    except Exception as e:
        print(f"[{'WSS' if ssl_ctx else 'WS'}] Bind failed on port {port}: {e}")
        return

    while True:
        try:
            sock, addr = server_sock.accept()
            if ssl_ctx:
                try:
                    sock = ssl_ctx.wrap_socket(sock, server_side=True)
                except Exception as ex:
                    try:
                        sock.close()
                    except Exception:
                        pass
                    continue
            t = threading.Thread(target=ws_client_worker, args=(sock, addr), daemon=True)
            t.start()
        except Exception:
            time.sleep(0.5)

class CustomHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC_DIR, **kwargs)

    def log_message(self, format, *args):
        # Silence excessive telemetry polling logs
        if len(args) > 0 and "/api/telemetry" in str(args[0]):
            return
        super().log_message(format, *args)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        if self.path in ("/api/ip", "/api/telemetry") or self.path.startswith("/api/nearby-images") or self.path.startswith("/api/gnn/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            return
        return super().do_HEAD()

    def do_GET(self):
        if self.path.startswith("/api/nearby-images"):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                lon = float(qs.get("lon", [77.4526])[0])
                lat = float(qs.get("lat", [18.4449])[0])
                count = int(qs.get("count", [8])[0])
            except Exception:
                lon, lat, count = 77.4526, 18.4449, 8

            images = find_nearby_images(lon, lat, count=count)
            payload = {
                "status": "success",
                "count": len(images),
                "query": {"lon": lon, "lat": lat},
                "images": images
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # High-reliability image proxy to bypass CORS / hotlink / 403 blocks on raw images
        if self.path.startswith("/api/image-proxy"):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            target_url = qs.get("url", [""])[0]
            if not target_url:
                self.send_response(400)
                self.end_headers()
                return
            try:
                req = urllib.request.Request(target_url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Referer": "https://mars.nasa.gov/mars2020/multimedia/raw-images/",
                    "Accept": "image/*,*/*;q=0.8"
                })
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = resp.read()
                    ctype = resp.headers.get("Content-Type", "image/jpeg")
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=604800")
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception as e:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Image proxy failure: {e}".encode())
                return

        # GNN Health Check Proxy (Bypasses HTTPS Mixed-Content block and enables LAN remote clients)
        if self.path.startswith("/api/gnn/health"):
            try:
                req = urllib.request.Request("http://127.0.0.1:8023/health", headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=1.2) as r:
                    gnn_body = r.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(gnn_body)))
                self.end_headers()
                self.wfile.write(gnn_body)
                return
            except Exception as e:
                offline_payload = json.dumps({
                    "status": "offline",
                    "error": str(e),
                    "backend": "Classical Sobel/Laplacian Offline Fallback",
                    "active": False
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(offline_payload)))
                self.end_headers()
                self.wfile.write(offline_payload)
                return
        if self.path == "/api/ip":
            local_ip = get_local_ip()
            payload = {
                "ip": local_ip,
                "http_port": HTTP_PORT,
                "https_port": HTTPS_PORT,
                "ws_port": WS_PORT,
                "wss_port": WSS_PORT,
                "phone_url_https": f"https://{local_ip}:{HTTPS_PORT}/phone.html",
                "phone_url_http": f"http://{local_ip}:{HTTP_PORT}/phone.html",
                "phone_url": f"https://{local_ip}:{HTTPS_PORT}/phone.html", # default to HTTPS for gyro
                "ws_url": f"ws://{local_ip}:{WS_PORT}",
                "wss_url": f"wss://{local_ip}:{WSS_PORT}"
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/telemetry"):
            with state_lock:
                merged = {
                    **latest_phone_telemetry,
                    "phone": latest_phone_telemetry,
                    "pc": latest_pc_telemetry,
                    "server_time": time.time()
                }
                body = json.dumps(merged).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/telemetry":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            try:
                data = json.loads(raw_body.decode("utf-8"))
                msg_type = data.get("type", "phone_input")
                with state_lock:
                    if msg_type == "phone_input":
                        latest_phone_telemetry.update(data)
                        latest_phone_telemetry["updated_at"] = time.time()
                    elif msg_type == "pc_telemetry":
                        latest_pc_telemetry.update(data)
                        latest_pc_telemetry["updated_at"] = time.time()
                    else:
                        latest_phone_telemetry.update(data)
                        latest_phone_telemetry["updated_at"] = time.time()
                broadcast_ws(json.dumps(data))
                resp = json.dumps({"status": "ok", "synced": True, "server_time": time.time()}).encode("utf-8")
            except Exception as e:
                resp = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        # GNN Inference Proxy (Forwards payload to GNN service on 8023)
        if self.path.startswith("/api/gnn/predict-graph"):
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:8023/predict-graph",
                    data=raw_body,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=3.0) as r:
                    resp_data = r.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)
                return
            except Exception as e:
                err_resp = json.dumps({"error": f"GNN backend error: {e}"}).encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_resp)))
                self.end_headers()
                self.wfile.write(err_resp)
                return

        self.send_response(404)
        self.end_headers()

class ReuseHTTPServer(HTTPServer):
    allow_reuse_address = True

def main():
    local_ip = get_local_ip()
    print("=" * 70)
    print("🚀 NASA MARS 2020 EVA TELEMETRY & WEB SERVER (ALL FUNCTIONALITIES)")
    print(f"📡 HTTP Web:        http://{local_ip}:{HTTP_PORT}/")
    print(f"🔒 HTTPS Web:       https://{local_ip}:{HTTPS_PORT}/ (Recommended for Gyro)")
    print(f"📱 Phone Web App:   https://{local_ip}:{HTTPS_PORT}/phone.html")
    print(f"⚡ WebSocket Relay: ws://{local_ip}:{WS_PORT}")
    print(f"⚡ Secure WS Relay: wss://{local_ip}:{WSS_PORT}")
    print(f"🧠 AI GNN Backend:  http://127.0.0.1:8023 (L40 GPU Bioenergetics)")
    print(f"📂 Static Root:     {PUBLIC_DIR}")
    print("=" * 70)

    # 0. Check and auto-start AI GNN L40 Backend Server on port 8023 if not running
    def start_gnn_server():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(("127.0.0.1", 8023))
            s.close()
            print("[GNN] AI GNN Backend already running on port 8023.")
        except Exception:
            try:
                import gnn_l40_server
                gnn_thread = threading.Thread(target=gnn_l40_server.main, daemon=True)
                gnn_thread.start()
                print("[GNN] AI GNN L40 Backend Server auto-started on http://127.0.0.1:8023")
            except Exception as e:
                print(f"[GNN] Could not auto-start GNN server: {e}")

    start_gnn_server()

    # 1. Start standard WS server
    ws_thread = threading.Thread(target=run_ws_server, args=(WS_PORT, None), daemon=True)
    ws_thread.start()

    # 2. Check for SSL Cert & Key for HTTPS & WSS
    ssl_ctx = None
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        try:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
            print("[SSL] TLS 1.3/1.2 Certificate loaded successfully.")

            # Start WSS server
            wss_thread = threading.Thread(target=run_ws_server, args=(WSS_PORT, ssl_ctx), daemon=True)
            wss_thread.start()

            # Start HTTPS server in background thread
            def run_https():
                httpsd = ReuseHTTPServer(("0.0.0.0", HTTPS_PORT), CustomHTTPHandler)
                httpsd.socket = ssl_ctx.wrap_socket(httpsd.socket, server_side=True)
                print(f"[HTTPS] Serving secure web on https://0.0.0.0:{HTTPS_PORT}")
                httpsd.serve_forever()

            https_thread = threading.Thread(target=run_https, daemon=True)
            https_thread.start()
        except Exception as e:
            print(f"[SSL] Warning: Could not initialize HTTPS server: {e}")
    else:
        print("[SSL] cert.pem or key.pem not found, running HTTP-only mode.")

    # 3. Start main HTTP server on port 8080
    httpd = ReuseHTTPServer(("0.0.0.0", HTTP_PORT), CustomHTTPHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    main()
