#!/usr/bin/env python3


import glob
import http.server
import socketserver
from argparse import ArgumentParser


class MyHTTPHandler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/!list':  # provide newline-separated listing of JSON files in the current directory
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            response = "\n".join(glob.glob("*.json"))
            self.wfile.write(response.encode())
        else:  # default: provide the file
            super().do_GET()


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('port', nargs='?', default=8000, type=int)

    args = ap.parse_args()

    with socketserver.TCPServer(("", args.port), MyHTTPHandler) as httpd:
        print("serving at port", args.port)
        httpd.serve_forever()
