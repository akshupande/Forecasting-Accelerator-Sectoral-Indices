import http.server
import socketserver
import webbrowser
import os
import socket
from pathlib import Path

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def start_dashboard_server(port=8080, output_dir="output"):
    
    dashboard_dir = Path(output_dir) / "dashboard"
    if not dashboard_dir.exists():
        print("[ERROR] Dashboard directory not found: {}".format(dashboard_dir))
        print("Please run the forecasting script first to generate the dashboard.")
        return
    
    os.chdir(dashboard_dir)
    
    handler = http.server.SimpleHTTPRequestHandler
    
    local_ip = get_local_ip()
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print("[DASHBOARD] Server started!")
        print("   Local access: http://localhost:{}".format(port))
        print("   Network access: http://{}:{}".format(local_ip, port))
        print("   Serving from: {}".format(dashboard_dir))
        print("   http://{}:{}".format(local_ip, port))
        print("\nPress Ctrl+C to stop the server")
        
        webbrowser.open("http://localhost:{}".format(port))
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    start_dashboard_server(port=8080, output_dir="output")