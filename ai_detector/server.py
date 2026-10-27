"""Tiny local web UI (stdlib only)."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from .detector import analyze

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>AI Detector</title>
<style>body{font:16px system-ui;max-width:820px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;height:220px;padding:.6rem;box-sizing:border-box}
button{padding:.6rem 1.2rem;font-size:1rem;margin-top:.5rem;cursor:pointer}
.score{font-size:2.2rem;font-weight:700;margin:1rem 0 .2rem}
.bar{height:14px;border-radius:7px;background:#ddd;overflow:hidden}.bar>div{height:100%}
#text span{padding:1px 0}.note{color:#666;font-size:.85rem;margin-top:1.5rem}
td{padding:.1rem .8rem .1rem 0}</style>
<h1>AI Detector</h1>
<textarea id=in placeholder="Paste text (40+ words)…"></textarea><br>
<button onclick=run()>Analyze</button>
<div id=out></div>
<p class=note>Heuristic stylometric analysis, not a definitive test. Like all AI detectors it can produce
false positives (especially for non-native English writers) and must not be used as sole evidence of misconduct.</p>
<script>
async function run(){
 const r=await (await fetch('/analyze',{method:'POST',body:document.getElementById('in').value})).json();
 const c=r.score>=70?'#d33':r.score>=40?'#e90':'#2a2';
 let h=`<div class=score style="color:${c}">${r.score}% AI</div><div>${r.label} · ${r.word_count} words</div>
 <div class=bar><div style="width:${r.score}%;background:${c}"></div></div><h3>Signals</h3><table>`;
 for(const k in r.signals)h+=`<tr><td>${k.replace(/_/g,' ')}</td><td>${r.signals[k]}</td></tr>`;
 h+='</table><h3>Sentence highlights</h3><div id=text>';
 for(const [s,sc] of r.sentences){const a=Math.max(0,(sc-40)/60);
  h+=`<span title="${sc}%" style="background:rgba(221,51,51,${a.toFixed(2)})">${s.replace(/</g,'&lt;')}</span> `}
 document.getElementById('out').innerHTML=h+'</div>';}
</script>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send(200, PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        r = analyze(self.rfile.read(min(n, 1_000_000)).decode("utf-8", "replace"))
        self._send(200, json.dumps({"score": r.score, "label": r.label, "signals": r.signals,
                                    "sentences": r.sentences, "word_count": r.word_count}).encode(),
                   "application/json")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def serve(port=8000):
    print(f"Open http://localhost:{port}")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
