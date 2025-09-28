import webbrowser
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
from boxsdk import OAuth2
import os
from dotenv import load_dotenv

def get_new_refresh_token():
    """
    Opens a browser for the user to authenticate and returns a new refresh token.
    """
    load_dotenv()
    CLIENT_ID = os.getenv('BOX_CLIENT_ID')
    CLIENT_SECRET = os.getenv('BOX_CLIENT_SECRET')
    REDIRECT_URI = 'http://localhost:8080'

    HOST, PORT = '127.0.0.1', 8080
    auth_code = None

    # Initialize OAuth
    oauth = OAuth2(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        store_tokens=None
    )

    # Get auth URL
    auth_url, csrf = oauth.get_authorization_url(REDIRECT_URI)
    webbrowser.open(auth_url)

    # Local server for callback
    class CallbackHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            nonlocal auth_code
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>Authenticated! You can close this tab.</h1>')
            
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if 'code' in query:
                auth_code = query['code'][0]
                print(f"Auth code received: {auth_code[:20]}...")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((HOST, PORT), CallbackHandler) as httpd:
        print(f'Local server running at: http://{HOST}:{PORT}')
        httpd.handle_request()

    # Exchange for tokens
    if auth_code:
        access_token, refresh_token = oauth.authenticate(auth_code)
        print(f"New Refresh Token: {refresh_token}")
        return refresh_token
    else:
        print("Authorization failed.")
        return None

if __name__ == '__main__':
    get_new_refresh_token()
