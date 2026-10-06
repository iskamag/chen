'use strict';
const $ = id => document.getElementById(id);
// Vector silhouettes avoid platform emoji fonts and keep both sides distinct.
const pieceShapes = {
  p: '<circle cx="32" cy="17" r="8"/><path d="M25 27h14l-3 7 3 13H25l3-13z"/>',
  n: '<path d="M21 47c0-9 6-13 11-19l-10 5-8-5 10-14 1-7 8 5c14 3 18 18 16 35z"/><path d="M24 20h3"/>',
  b: '<path d="M32 8c-3 5-11 10-11 17 0 5 4 8 8 9l-6 13h18l-6-13c4-1 8-4 8-9 0-7-8-12-11-17z"/><path d="m35 19-6 8"/>',
  r: '<path d="M18 10h7v7h5v-7h5v7h5v-7h7v14l-7 5 1 18H23l1-18-6-5z"/>',
  q: '<path d="m17 19 6 25h18l6-25-10 12-5-17-5 17z"/><circle cx="16" cy="16" r="3"/><circle cx="32" cy="11" r="3"/><circle cx="48" cy="16" r="3"/>',
  k: '<path d="M32 6v13m-6-7h12"/><path d="M25 44 20 29c-4-11 7-13 12-5 5-8 16-6 12 5l-5 15z"/>'
};
function pieceIcon(piece) {
  // An image document isolates these semantic colors from page recoloring.
  const white=piece===piece.toUpperCase();
  const image=document.createElement('img');
  image.className='piece '+(white ? 'white':'black');
  image.alt='';image.setAttribute('aria-hidden', 'true');image.draggable=false;
  const svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" style="color-scheme:only light"><g fill="'+(white ? '#fff8e9':'#243e31')+'" stroke="'+(white ? '#3f4e42':'#dbe6d7')+'" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'+pieceShapes[piece.toLowerCase()]+'<path d="M22 47h20l4 7H18z"/><path d="M17 54h30v4H17z"/></g></svg>';
  image.src='data:image/svg+xml,'+encodeURIComponent(svg);
  return image;
}
const names = {p:'pawn',n:'knight',b:'bishop',r:'rook',q:'queen',k:'king'};
let state, selected = null, flipped = false, busy = false, activeJob = null, generation = 0;
const squareName = n => 'abcdefgh'[n % 8] + (Math.floor(n / 8) + 1);
function setBusy(value) {
  busy = value; document.body.classList.toggle('busy', value);
  document.querySelectorAll('#setup button,#setup input,#setup select').forEach(el => el.disabled = value);
  $('thinking').textContent = value ? 'Waiting for Chen…' : 'Paws at the ready';
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
  return {backend:$('backend').value, delta_cp:Number($('delta').value), theta:Number($('theta').value), side:$('side').value, level:Number($('level').value), time_ms:Number($('time').value), book:$('book').checked,
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
  // Each response includes the human move and Chen's reply. Only a loss
  // of the human's pieces identifies Chen's capture (including en passant).
  const white=after.human==='white';
  const count=s=>s.pieces.filter(p=>p && (p===p.toUpperCase())===white).length;
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
      button.append(pieceIcon(piece));
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
  for(let i=0;i<state.moves.length;i+=2){const row=document.createElement('div');row.className='move-row';for(const text of [i/2+1+'.',state.san_moves?.[i]||state.moves[i],state.san_moves?.[i+1]||state.moves[i+1]||'']){const span=document.createElement('span');span.textContent=text;row.append(span);}$('moves').append(row);}
  $('moves').scrollTop=$('moves').scrollHeight;
  const t=state.telemetry;
  const horizon=t?.book ? (t.opening ? `Opening: ${t.opening}` : 'From the book') : (t?.drowsy ? `Explored ply ${t.drowsy.deepest_ply}` : `Depth ${t?.depth}`);
  $('telemetry').textContent=t ? `${horizon} · ${t.nodes.toLocaleString()} nodes · ${Math.round(t.nps/1000)}k nodes/s · ${t.ms} ms search · ${Math.round(state.queue_ms||0)} ms queue` : 'No search yet.';
  if(t?.drowsy) $('telemetry').textContent+=` · ${t.drowsy.fail_meow} fail-meows · ${t.drowsy.fail_zoomies} zoomies · ${t.drowsy.ignored_boring} ignored moves`;
  $('search-description').textContent=state.backend==='drowsy' ? 'Drowsy Chen: an experimental drifting-score, selective attention search toy. Strength is unmeasured. Search time excludes queue wait.' : 'Native Chen CPU search, tapered evaluation, PVS enabled. The opening book plays varied named lines for the first plies, then search takes over. The selected time starts when search runs; queue wait is separate. This is our playable development checkpoint.';
  $('revision').textContent='Build '+state.revision+' · '+(state.backend==='drowsy'?'Drowsy':'Serious')+' · resource level '+state.level;
}
function capitalize(s){return s[0].toUpperCase()+s.slice(1);}
async function promotion(moves) {
  $('promotion-choices').replaceChildren();
  for(const move of moves){const button=document.createElement('button');button.value=move;button.append(pieceIcon(state.human==='white' ? move[4].toUpperCase() : move[4]));button.setAttribute('aria-label','Promote to '+names[move[4]]);$('promotion-choices').append(button);}
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
    try{const next=await submit('/api/move',{backend:state.backend||'serious',delta_cp:state.delta_cp??80,theta:state.theta??0.55,book:!!state.book,side:state.human,level:state.level,time_ms:state.milliseconds,depth:state.depth_limit,nodes:state.node_limit,moves:state.moves,move,fen:state.fen});if(!next)return;state=next;save();react(before,state);}catch(error){$('status').textContent=error.message;setBusy(false);mood('idle','My paws are back. Your board is safe.');return;}
    setBusy(false);render();return;
  }
  selected=selected===name ? null : (state.legal.some(m=>m.startsWith(name)) ? name : null);render();
}
$('setup').onsubmit=async event=>{
  event.preventDefault();if(busy)return;setBusy(true);selected=null;
  try{const next=await submit('/api/new',limits());if(!next)return;state=next;flipped=state.human==='black';save();setBusy(false);react(null,state);render();}catch(error){setBusy(false);$('status').textContent=error.message;mood('idle','A tiny pause. Try again in a moment.');}
};
function drowsyControls(){ $('drowsy-controls').hidden=$('backend').value!=='drowsy'; }
$('backend').onchange=drowsyControls;
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
    const restored=!!(saved && Array.isArray(saved.moves) && saved.pieces?.length===64);
    state=restored ? saved : await request('/api/state');
    // Preserve games saved before the distracted backend was named Drowsy.
    if(state.backend==='catnip') state.backend='drowsy';
    if(state.telemetry?.catnip){state.telemetry.drowsy=state.telemetry.catnip;delete state.telemetry.catnip;}
    if(state.milliseconds===3000) state.milliseconds=1000;
    if(state.san_moves?.length!==state.moves.length){
      const replay=await request('/api/state',{moves:state.moves,side:state.human,level:state.level});
      state.san_moves=replay.san_moves;
    }
    save();
    $('backend').value=state.backend||'serious';$('delta').value=state.delta_cp??80;$('delta-value').textContent=$('delta').value;$('theta').value=state.theta??0.55;$('theta-value').textContent=Number($('theta').value).toFixed(2);drowsyControls();if(restored)$('book').checked=state.book!==false;
    flipped=state.human==='black';$('side').value=state.human;$('level').value=state.level;$('level-value').textContent=state.level;$('time').value=state.milliseconds;
    $('depth').value=state.depth_limit===64?'':state.depth_limit||'';$('nodes').value=state.node_limit||'';render();
  }catch(error){$('status').textContent='Could not connect: '+error.message;}
})();
