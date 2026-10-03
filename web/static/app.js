'use strict';
const $ = id => document.getElementById(id);
const glyphs = {P:'♟',N:'♞',B:'♝',R:'♜',Q:'♛',K:'♚',p:'♟',n:'♞',b:'♝',r:'♜',q:'♛',k:'♚'};
const names = {p:'pawn',n:'knight',b:'bishop',r:'rook',q:'queen',k:'king'};
let state, selected = null, flipped = false, busy = false, activeJob = null, generation = 0;
const squareName = n => 'abcdefgh'[n % 8] + (Math.floor(n / 8) + 1);
function setBusy(value) {
  busy = value; document.body.classList.toggle('busy', value);
  document.querySelectorAll('#setup button,#setup input,#setup select').forEach(el => el.disabled = value);
  $('thinking').textContent = value ? 'Waiting for Chen…' : 'Ready to pounce';
  $('cancel').hidden = !value;
  if(value) mood('thinking', 'A little wiggle while I think…');
}
async function request(path, data) {
  const endpoint = path.startsWith('/') ? '.'+path : path;
  const response = await fetch(endpoint, data === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}
function mood(name, line) {
  $('avatar').dataset.mood=name;
  $('chen-says').textContent=line;
}
function limits() {
  return {backend:$('backend').value, delta_cp:Number($('delta').value), theta:Number($('theta').value), side:$('side').value, level:Number($('level').value), time_ms:Number($('time').value),
    depth:$('depth').value ? Number($('depth').value) : null,
    nodes:$('nodes').value ? Number($('nodes').value) : null};
}
async function submit(path, data) {
  const ticket=++generation;
  try {
  const job=await request(path,data);
  if(ticket!==generation){await request('/api/cancel',{id:job.id});return null;}
  activeJob=job.id;
  const started=performance.now();
  while(ticket===generation){
    const status=await request('/api/jobs/'+job.id);
    if(ticket!==generation)return null;
    if(status.phase==='done'){activeJob=null;status.state.queue_ms=status.queue_ms;return status.state;}
    if(status.phase==='cancelled')throw new Error('Search cancelled. Your board is unchanged.');
    if(status.phase==='error')throw new Error(status.error);
    $('thinking').textContent=status.phase==='queued' ? `In the queue · ${(Number(status.waiting_ms||0)/1000).toFixed(1)}s` : 'Chen is thinking…';
    if(performance.now()-started>65000){await request('/api/cancel',{id:job.id});throw new Error('Search timed out. Please try again.');}
    await new Promise(resolve=>setTimeout(resolve,150));
  }
  return null;
  } catch(error) { if(ticket!==generation)return null; throw error; }
}
function react(before, after) {
  const count=s=>s.pieces.filter(Boolean).length;
  if(after.outcome){mood(after.outcome.includes('Chen wins')?'win':after.outcome.includes('You win')?'loss':'idle', after.outcome.includes('Chen wins')?'Nya! That was a lovely little hunt.':after.outcome.includes('You win')?'Mrrp. Rematch?':'A peaceful little truce.');}
  else if(after.check)mood('check','Check! Did you see that paw coming?');
  else if(before && count(after)<count(before))mood('pounce','Pounce! A little treasure for the tails.');
  else mood('idle','Your turn. I promise my tails are behaving.');
}
function save(){try{sessionStorage.setItem('chen-game',JSON.stringify(state));}catch{}}
function render() {
  if (!state) return;
  const focused=document.activeElement?.dataset.square;
  $('board').replaceChildren();
  const last = state.moves.at(-1);
  for (let row = 0; row < 8; row++) for (let column = 0; column < 8; column++) {
    const rank = flipped ? row : 7-row, file = flipped ? 7-column : column, n = rank*8+file;
    const name = squareName(n), piece = state.pieces[n];
    const button = document.createElement('button'); button.className = 'square'; button.dataset.square = name;
    if ((rank+file)%2 === 0) button.classList.add('dark');
    if (selected === name) button.classList.add('selected');
    if (selected && state.legal.some(m => m.startsWith(selected+name))) button.classList.add('target');
    if (last && [last.slice(0,2),last.slice(2,4)].includes(name)) button.classList.add('last');
    button.setAttribute('aria-label', name + (piece ? ', '+ (piece===piece.toUpperCase() ? 'white ' : 'black ') + names[piece.toLowerCase()] : ', empty'));
    if (piece) {
      const span = document.createElement('span'); span.className='piece '+(piece===piece.toUpperCase() ? 'white':'black');span.textContent=glyphs[piece];span.setAttribute('aria-hidden','true');button.append(span);
    }
    if(column===0){const span=document.createElement('span');span.className='coordinate rank';span.textContent=rank+1;button.append(span);}
    if(row===7){const span=document.createElement('span');span.className='coordinate file';span.textContent='abcdefgh'[file];button.append(span);}
    button.onclick = () => click(name); $('board').append(button);
  }
  if(focused)document.querySelector(`[data-square="${focused}"]`)?.focus({preventScroll:true});
  $('player').textContent='You · '+capitalize(state.human);
  $('opponent').textContent='Chen · '+(state.human==='white'?'Black':'White');
  $('status').textContent=state.outcome || (state.side===state.human ? 'Your move'+(state.check?' — check!':'.') : 'Chen to move.');
  $('moves').replaceChildren();
  if (!state.moves.length) $('moves').textContent='The board is all yours.';
  for(let i=0;i<state.moves.length;i+=2){const row=document.createElement('div');row.className='move-row';for(const text of [i/2+1+'.',state.moves[i],state.moves[i+1]||'']){const span=document.createElement('span');span.textContent=text;row.append(span);}$('moves').append(row);}
  $('moves').scrollTop=$('moves').scrollHeight;
  const t=state.telemetry;
  const horizon=t?.catnip ? `Explored ply ${t.catnip.deepest_ply}` : `Depth ${t?.depth}`;
  $('telemetry').textContent=t ? `${horizon} · ${t.nodes.toLocaleString()} nodes · ${Math.round(t.nps/1000)}k nodes/s · ${t.ms} ms search · ${Math.round(state.queue_ms||0)} ms queue` : 'No search yet.';
  if(t?.catnip) $('telemetry').textContent+=` · ${t.catnip.fail_meow} fail-meows · ${t.catnip.fail_zoomies} zoomies · ${t.catnip.ignored_boring} ignored moves`;
  $('search-description').textContent=state.backend==='catnip' ? 'Catnip Chen: an experimental drifting-score, selective attention search toy. Strength is unmeasured. Search time excludes queue wait.' : 'Native Chen CPU search, tapered evaluation, PVS enabled. The selected time starts when search runs; queue wait is separate. This is our playable development checkpoint.';
  $('revision').textContent='Build '+state.revision+' · '+(state.backend==='catnip'?'Catnip':'Serious')+' · resource level '+state.level;
}
function capitalize(s){return s[0].toUpperCase()+s.slice(1);}
async function promotion(moves) {
  $('promotion-choices').replaceChildren();
  for(const move of moves){const button=document.createElement('button');button.value=move;button.textContent=glyphs[move[4]];button.setAttribute('aria-label','Promote to '+names[move[4]]);$('promotion-choices').append(button);}
  $('promotion').returnValue='cancel';$('promotion').showModal();
  return new Promise(resolve=>$('promotion').addEventListener('close',()=>resolve($('promotion').returnValue),{once:true}));
}
async function click(name) {
  if(busy || !state || state.outcome || state.side!==state.human)return;
  const moves=selected ? state.legal.filter(m=>m.startsWith(selected+name)) : [];
  if(moves.length){
    const move=moves.length>1 ? await promotion(moves) : moves[0];
    if(move==='cancel')return;
    selected=null;setBusy(true);
    const before=state;
    try{const next=await submit('/api/move',{backend:state.backend||'serious',delta_cp:state.delta_cp??80,theta:state.theta??0.55,side:state.human,level:state.level,time_ms:state.milliseconds,depth:state.depth_limit,nodes:state.node_limit,moves:state.moves,move,fen:state.fen});if(!next)return;state=next;save();react(before,state);}catch(error){$('status').textContent=error.message;setBusy(false);mood('idle','My paws are back. Your board is safe.');return;}
    setBusy(false);render();return;
  }
  selected=selected===name ? null : (state.legal.some(m=>m.startsWith(name)) ? name : null);render();
}
$('setup').onsubmit=async event=>{
  event.preventDefault();if(busy)return;setBusy(true);selected=null;
  try{const next=await submit('/api/new',limits());if(!next)return;state=next;flipped=state.human==='black';save();setBusy(false);react(null,state);render();}catch(error){setBusy(false);$('status').textContent=error.message;mood('idle','A tiny pause. Try again in a moment.');}
};
function catnipControls(){ $('catnip-controls').hidden=$('backend').value!=='catnip'; }
$('backend').onchange=catnipControls;
$('delta').oninput=()=>{$('delta-value').textContent=$('delta').value;};
$('theta').oninput=()=>{$('theta-value').textContent=Number($('theta').value).toFixed(2);};
$('level').oninput=()=>{$('level-value').textContent=$('level').value;};
$('flip').onclick=()=>{flipped=!flipped;render();};
$('copy').onclick=async()=>{try{await navigator.clipboard.writeText(state.fen);$('copy').textContent='FEN copied ✓';setTimeout(()=>{$('copy').textContent='Copy current FEN';},1800);}catch{$('status').textContent=state.fen;}};
$('cancel').onclick=async()=>{
  generation++;const id=activeJob;activeJob=null;setBusy(false);
  $('status').textContent='Search cancelled. Your board is unchanged.';mood('idle','A little breather. We can pounce later.');
  if(id)try{await request('/api/cancel',{id});}catch{}
};
fetch('./chen.svg').then(response=>response.text()).then(svg=>{$('avatar').innerHTML=svg;});
(async()=>{
  try{
    let saved;try{saved=JSON.parse(sessionStorage.getItem('chen-game'));}catch{}
    state=saved && Array.isArray(saved.moves) && saved.pieces?.length===64 ? saved : await request('/api/state');
    $('backend').value=state.backend||'serious';$('delta').value=state.delta_cp??80;$('delta-value').textContent=$('delta').value;$('theta').value=state.theta??0.55;$('theta-value').textContent=Number($('theta').value).toFixed(2);catnipControls();
    flipped=state.human==='black';$('side').value=state.human;$('level').value=state.level;$('level-value').textContent=state.level;$('time').value=state.milliseconds;
    $('depth').value=state.depth_limit===64?'':state.depth_limit||'';$('nodes').value=state.node_limit||'';render();
  }catch(error){$('status').textContent='Could not connect: '+error.message;}
})();
