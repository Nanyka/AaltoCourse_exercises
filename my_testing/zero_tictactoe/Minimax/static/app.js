
const boardEl = document.getElementById('board');
const invXEl  = document.getElementById('invX');
const invOEl  = document.getElementById('invO');
const turnEl  = document.getElementById('turn');
const statusEl = document.getElementById('status');
const sideX = document.getElementById('sideX');
const sideO = document.getElementById('sideO');
const newGameBtn = document.getElementById('newGame');
const pieces = Array.from(document.querySelectorAll('.piece'));
let selectedV = null;
let humanSide = 'X';
let state = null;
let legal = [];

function fmtInv(inv){ return `1×${inv[0]} 2×${inv[1]} 3×${inv[2]}`; }
function setStatus(text){ statusEl.textContent = text; }

function renderBoard(board){
  boardEl.innerHTML = '';
  for (let i=0;i<9;i++){
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.dataset.i = i;
    const c = board[i];
    if (c !== 0){
      const owner = [0,'x','x','x','o','o','o'][c];
      const val = [0,1,2,3,1,2,3][c];
      const tile = document.createElement('div');
      tile.className = `tile ${owner}`;
      tile.textContent = (owner==='x'?'X':'O') + val;
      cell.appendChild(tile);
    }
    cell.addEventListener('click', onCellClick);
    boardEl.appendChild(cell);
  }
}

function isLegal(v,i){ return legal.some(m => m.v===v && m.i===i); }

async function api(path, body){
  const res = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body||{})});
  if (!res.ok){
    let msg = await res.text();
    try{ const j=JSON.parse(msg); msg=j.detail || msg; }catch(_){}
    throw new Error(msg);
  }
  return await res.json();
}

function updateUI(payload){
  state = payload.state;
  legal = payload.legal || [];
  renderBoard(state.board);
  invXEl.textContent = 'X inv: ' + fmtInv(state.invX);
  invOEl.textContent = 'O inv: ' + fmtInv(state.invO);
  turnEl.textContent = 'turn: ' + state.to_move;
  const res = payload.result;
  if (res && res!=='ongoing'){ setStatus(res==='draw' ? 'Draw!' : `${res} wins!`); confetti(); }
  else setStatus(state.to_move===humanSide ? 'Your move' : 'Bot thinking…');
}

async function startNew(){
  const payload = await api('/api/new_game', {human_side: humanSide});
  selectedV = null; pieces.forEach(p=>p.classList.remove('active'));
  updateUI(payload);
  if (state.to_move !== humanSide) setTimeout(botMove, 400);
}

async function humanMove(v,i){
  if (!isLegal(v,i)){ setStatus('Illegal move.'); return; }
  const payload = await api('/api/human_move', {v,i});
  updateUI(payload);
  if (payload.result === 'ongoing') setTimeout(botMove, 450);
}

async function botMove(){
  const payload = await api('/api/bot_move', {});
  updateUI(payload);
}

function onCellClick(e){
  const i = parseInt(e.currentTarget.dataset.i,10);
  if (selectedV===null){ setStatus('Pick a piece value first.'); return; }
  humanMove(selectedV, i);
}

pieces.forEach(p => {
  p.addEventListener('click', () => {
    pieces.forEach(q => q.classList.remove('active'));
    p.classList.add('active');
    selectedV = parseInt(p.dataset.v,10);
    setStatus('Selected piece: ' + selectedV);
  });
});

sideX.addEventListener('click', () => { sideX.classList.add('active'); sideO.classList.remove('active'); humanSide='X'; });
sideO.addEventListener('click', () => { sideO.classList.add('active'); sideX.classList.remove('active'); humanSide='O'; });
newGameBtn.addEventListener('click', startNew);

function confetti(){
  const cvs = document.getElementById('confetti');
  const ctx = cvs.getContext('2d');
  const W = cvs.width = innerWidth, H = cvs.height = innerHeight;
  const N = 120, parts = [];
  for (let k=0;k<N;k++){
    parts.push({x:Math.random()*W, y:-10, vy:2+Math.random()*3, vx:(Math.random()-.5)*2, r:2+Math.random()*3, a:Math.random()*Math.PI, col:Math.random()<.5?'#7c9cff':'#77e0c6'});
  }
  let t=0; (function step(){ t++; ctx.clearRect(0,0,W,H);
    parts.forEach(p=>{ p.x+=p.vx; p.y+=p.vy; p.a+=.1; ctx.save(); ctx.translate(p.x,p.y); ctx.rotate(p.a); ctx.fillStyle=p.col; ctx.fillRect(-p.r,-p.r,2*p.r,2*p.r); ctx.restore(); });
    if (t<120) requestAnimationFrame(step);
  })();
}
startNew();
