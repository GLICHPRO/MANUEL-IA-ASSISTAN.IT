"""
Simple HTTP Server per frontend Gideon 2.0
"""
import http.server
import socketserver
import os

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Aggiungi headers CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", PORT), MyHTTPRequestHandler)
    
    print(f"╔════════════════════════════════════════════╗")
    print(f"║  Gideon 2.0 Frontend Server               ║")
    print(f"╚════════════════════════════════════════════╝")
    print(f"\n🌐 Server attivo su: http://localhost:{PORT}")
    print(f"📱 Apri il browser a: http://localhost:{PORT}/index.html")
    print(f"\n✓ WebSocket funzionerà correttamente")
    print(f"\nPremi CTRL+C per fermare il server\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server fermato")
    finally:
        httpd.server_close()
