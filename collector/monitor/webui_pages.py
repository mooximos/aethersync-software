"""HTML for the web UI (Jinja templates). Plain HTML, CSS and JavaScript: no internet needed."""

BASE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{{ title }} - Intersection Monitor</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f5f7;color:#111}
nav{background:#1f3a5f;padding:8px;display:flex;flex-wrap:wrap;gap:6px}
nav a{color:#fff;text-decoration:none;padding:6px 10px;border-radius:6px;background:#2d5288}
main{padding:12px;max-width:1000px;margin:auto}
table{border-collapse:collapse;width:100%;background:#fff}td,th{border:1px solid #ccc;padding:6px;text-align:left}
button{font-size:1rem;padding:10px 14px;border-radius:8px;border:0;background:#2d5288;color:#fff}
input,select{font-size:1rem;padding:6px;max-width:100%}.big{font-size:1.4rem;font-weight:bold}
img.cam{width:100%;max-width:960px;display:block;margin-bottom:10px;background:#000}
.ok{color:#08702a}.bad{color:#b00020}
</style></head><body>
<nav><a href="/">Status</a><a href="/live">Live view</a><a href="/calibrate">Calibrate</a>
<a href="/download">Download</a><a href="/clock">Clock sync</a></nav>
<main><h2>{{ title }}</h2>{{ body|safe }}</main></body></html>"""

STATUS = """<p class="big" id="rec">...</p>
<p class="big bad" id="clockwarn" hidden></p>
<table id="tab"></table>
<p><button onclick="awake()">Stay awake for 1 hour</button> <span id="awake"></span></p>
<script>
function row(k,v){return '<tr><th>'+k+'</th><td>'+(v===null||v===''?'-':v)+'</td></tr>';}
async function load(){
 const s=await (await fetch('/api/status')).json();
 document.getElementById('rec').textContent=s.recording?'RECORDING':'Idle (not recording)';
 document.getElementById('rec').className='big '+(s.recording?'ok':'');
 const cw=document.getElementById('clockwarn'); cw.hidden=!s.clock_warning;
 cw.innerHTML=s.clock_warning?'NOT RECORDING: '+s.clock_warning+'. <a href="/clock">Sync the clock now</a>.':'';
 const phone=Date.now(), diff=(s.pi_time_ms-phone)/1000;
 let h=row('Next window',s.next_window)+row('Battery',s.battery_v===null?'-':s.battery_v.toFixed(2)+' V, '+s.power_w+' W')
 +row('CPU / Hailo temperature',s.cpu_temp_c+' C / '+s.hailo_temp_c+' C')
 +row('Frames per second (target '+s.fps_target+')',Object.entries(s.fps).map(e=>e[0]+': '+e[1]).join(', '))
 +row('Vehicles in the last 15 min',Object.entries(s.vehicles_15min).map(e=>e[0]+': '+e[1]).join(', '))
 +row('Disk free',s.disk_free_gb+' GB')+row('Night mode',s.night_mode?'on':'off')
 +row('Pi time minus phone time',diff.toFixed(1)+' s'+(Math.abs(diff)>0.5?' - please do a clock sync':''));
 document.getElementById('tab').innerHTML=h;
 document.getElementById('awake').textContent=s.stay_awake_until?'awake until '+s.stay_awake_until:'';
}
async function awake(){await fetch('/api/stay_awake',{method:'POST'});load();}
load();setInterval(load,5000);
</script>"""

LIVE = """<p>Pictures are made in memory and never saved. Lane lines: cyan, stop line: red,
detector zone: orange, view edge: gray, tiles: purple.</p>
{% for c in cameras %}<h3>Camera {{ c }}</h3><img class="cam" src="/stream/{{ c }}" alt="camera {{ c }}">{% endfor %}"""

CALIBRATE = """<p>1. Pick a camera. 2. Tap at least 6 road points you have measured (paint marks, lane line ends).
3. Type each point's x_m (meters from the curb toward the center line) and y_m (meters upstream of the stop line).
4. Check, then Save (only allowed when the mean error is below 0.5 m).</p>
<select id="cam">{% for c in cameras %}<option>{{ c }}</option>{% endfor %}</select>
<button onclick="reload()">Get picture</button>
<div style="position:relative"><img class="cam" id="pic" onclick="tap(event)"><div id="marks"></div></div>
<table id="pts"></table>
<p>lane_x_m (JSON, optional): <input id="lanes" size="40" placeholder='{"1": [0, 3.5], "2": [3.5, 7]}'></p>
<p>view_length_m (optional): <input id="view" size="6"> tiles (JSON, optional): <input id="tiles" size="30" placeholder="[[640,290,640,360]]"></p>
<p><button onclick="send(false)">Check</button> <button onclick="send(true)">Save</button></p><p id="out"></p>
<script>
let pts=[];
function reload(){pts=[];draw();document.getElementById('pic').src='/snapshot/'+cam.value+'.jpg?overlay=0&t='+Date.now();}
function tap(e){const im=e.target,r=im.getBoundingClientRect();
 const u=(e.clientX-r.left)*im.naturalWidth/r.width,v=(e.clientY-r.top)*im.naturalHeight/r.height;
 pts.push({px:[Math.round(u),Math.round(v)],m:['','']});draw();}
function draw(){let h='<tr><th>#</th><th>pixel</th><th>x_m</th><th>y_m</th><th>error m</th><th></th></tr>';
 pts.forEach((p,i)=>{h+='<tr><td>'+(i+1)+'</td><td>'+p.px+'</td><td><input size=5 value="'+p.m[0]+'" onchange="pts['+i+'].m[0]=this.value"></td>'
 +'<td><input size=5 value="'+p.m[1]+'" onchange="pts['+i+'].m[1]=this.value"></td><td id="e'+i+'"></td>'
 +'<td><button onclick="pts.splice('+i+',1);draw()">x</button></td></tr>';});
 document.getElementById('pts').innerHTML=h;}
async function send(save){
 const body={points:pts.map(p=>({px:p.px,m:[parseFloat(p.m[0]),parseFloat(p.m[1])]})),save:save};
 try{if(lanes.value)body.lane_x_m=JSON.parse(lanes.value);if(tiles.value)body.tiles=JSON.parse(tiles.value);}
 catch(e){out.textContent='lane_x_m or tiles is not valid JSON';return;}
 if(view.value)body.view_length_m=parseFloat(view.value);
 const r=await (await fetch('/api/calibrate/'+cam.value,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
 if(!r.ok){out.textContent=r.message;return;}
 r.errors_m.forEach((e,i)=>{document.getElementById('e'+i).textContent=e;});
 out.innerHTML='Mean error: <b>'+r.mean_error_m+' m</b>. '+(r.saved?'<span class=ok>Saved.</span> ':'')+r.message;}
</script>"""

DOWNLOAD = """<table><tr><th>Day</th><th>Size</th><th>Downloaded</th><th></th></tr>
{% for d in days %}<tr><td>{{ d.name }}</td><td>{{ d.size_mb }} MB</td><td>{{ d.downloaded or '-' }}</td>
<td><a href="/download/{{ d.name }}.zip">Download zip</a></td></tr>{% endfor %}</table>
{% if not days %}<p>No data yet.</p>{% endif %}"""

CLOCK = """<p>Sets the Pi's clock from this phone or laptop (only if they differ by more than 0.5 s).</p>
<p><button onclick="sync()">Sync clock now</button></p><p id="out"></p>
<script>
async function sync(){const r=await (await fetch('/api/clock',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({phone_ms:Date.now()})})).json();
 out.textContent=(r.changed?'Clock set. ':'No change needed. ')+'Difference was '+r.difference_s+' s.'
 +(r.problems&&r.problems.length?' Problems: '+r.problems.join('; '):'');}
</script>"""
