"""Drawing pad for collecting your own handwritten digits.

Run:  .venv/bin/python draw.py   then open http://localhost:8000
Each saved drawing goes to my_digits/<digit>/<n>.png
"""
import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

OUT = Path(__file__).parent / "my_digits"

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Digit Pad</title>
<style>
 body{font-family:system-ui;background:#111;color:#eee;display:flex;flex-direction:column;align-items:center;gap:12px;padding:20px}
 canvas{background:#000;border:2px solid #555;border-radius:8px;cursor:crosshair;touch-action:none}
 .row{display:flex;gap:6px;flex-wrap:wrap;justify-content:center}
 button{font-size:20px;padding:10px 14px;border-radius:6px;border:1px solid #555;background:#222;color:#eee;cursor:pointer}
 button:hover{background:#333} small{color:#999}
</style></head><body>
<h2>Draw one digit, then click its number to save</h2>
<canvas id="c" width="280" height="280"></canvas>
<div class="row" id="btns"></div>
<button onclick="clearPad()">Clear</button>
<div id="status"><small>Aim for ~10 of each digit. Vary your style a little.</small></div>
<div id="counts"></div>
<script>
const c=document.getElementById('c'),x=c.getContext('2d');
x.lineWidth=22;x.lineCap='round';x.lineJoin='round';x.strokeStyle='#fff';
let down=false;
function pos(e){const r=c.getBoundingClientRect();return[e.clientX-r.left,e.clientY-r.top]}
c.onpointerdown=e=>{down=true;x.beginPath();x.moveTo(...pos(e))};
c.onpointermove=e=>{if(down){x.lineTo(...pos(e));x.stroke()}};
window.onpointerup=()=>down=false;
function clearPad(){x.clearRect(0,0,280,280)}
async function save(d){
  const r=await fetch('/save',{method:'POST',body:JSON.stringify({label:d,png:c.toDataURL('image/png')})});
  const j=await r.json();document.getElementById('status').textContent='Saved a '+d+' → '+j.file;
  showCounts(j.counts);clearPad();
}
function showCounts(k){document.getElementById('counts').innerHTML='<small>'+
  [...Array(10).keys()].map(d=>d+': '+(k[d]||0)).join(' · ')+'</small>'}
for(let d=0;d<10;d++){const b=document.createElement('button');b.textContent=d;b.onclick=()=>save(d);btns.appendChild(b)}
fetch('/counts').then(r=>r.json()).then(showCounts);
</script></body></html>"""


def counts():
    return {d: len(list((OUT / str(d)).glob("*.png"))) for d in range(10)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        if self.path == "/counts":
            self._send(json.dumps(counts()), "application/json")
        else:
            self._send(PAGE, "text/html")

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        label = int(data["label"])
        if not 0 <= label <= 9:
            self.send_error(400, "label must be 0-9")
            return
        folder = OUT / str(label)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{len(list(folder.glob('*.png'))):03d}.png"
        path.write_bytes(base64.b64decode(data["png"].split(",", 1)[1]))
        self._send(json.dumps({"file": str(path.relative_to(OUT.parent)), "counts": counts()}),
                   "application/json")

    def log_message(self, *args):
        pass  # keep the terminal quiet


if __name__ == "__main__":
    print("Drawing pad running at http://localhost:8000  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
