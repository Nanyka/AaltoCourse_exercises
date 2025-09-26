// Redesigned UI logic
const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const sideX = document.getElementById('sideX');
const sideO = document.getElementById('sideO');
const newGameBtn = document.getElementById('newGame');

const trayPlayer = document.getElementById('trayPlayer');
const trayBot = document.getElementById('trayBot');
const playerBadge = document.getElementById('playerBadge');
const botBadge = document.getElementById('botBadge');

let selectedV = null;
let humanSide = 'X';
let state = null;
let legal = [];

function setStatus(t){ statusEl.textContent = t; }

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

function updatePieceDocks(){
  playerBadge.textContent = humanSide;
  playerBadge.className = 'badge ' + (humanSide==='X'?'x':'o');
  botBadge.textContent = humanSide==='X' ? 'O' : 'X';
  botBadge.className = 'badge ' + (humanSide==='X'?'o':'x');

  if (!state) return;
  const humanInv = (humanSide==='X') ? state.invX : state.invO;
  const botInv   = (humanSide==='X') ? state.invO : state.invX;
  const humanTurn = state.to_move === humanSide;

  function mkBtn(side, v, count, interactive){
    const btn = document.createElement(interactive ? 'button' : 'div');
    btn.className = (interactive ? 'pbtn ' : 'pbtn-ghost ') + (side==='X'?'x':'o');
    btn.dataset.v = v;
    btn.innerHTML = `<div class="pv">${v}</div><div class="count">${count}</div>`;
    if (interactive){
      if (count <= 0) {
        btn.classList.add('disabled'); btn.disabled = true;
      } else {
        const hasLegal = legal.some(m => m.v === v);
        if (!hasLegal || !humanTurn){ btn.classList.add('disabled'); btn.disabled = true; }
      }
      btn.addEventListener('click', () => {
        if (btn.classList.contains('disabled')) return;
        trayPlayer.querySelectorAll('.pbtn').forEach(b=>b.classList.remove('active'));
        btn.classList.add('active');
        selectedV = v;
        setStatus('Selected: '+humanSide+v);
      });
    } else {
      if (count <= 0) btn.classList.add('disabled');
    }
    return btn;
  }

  trayPlayer.innerHTML = '';
  trayPlayer.appendChild(mkBtn(humanSide, 1, humanInv[0], true));
  trayPlayer.appendChild(mkBtn(humanSide, 2, humanInv[1], true));
  trayPlayer.appendChild(mkBtn(humanSide, 3, humanInv[2], true));

  trayBot.innerHTML = '';
  const botSide = (humanSide==='X') ? 'O' : 'X';
  trayBot.appendChild(mkBtn(botSide, 1, botInv[0], false));
  trayBot.appendChild(mkBtn(botSide, 2, botInv[1], false));
  trayBot.appendChild(mkBtn(botSide, 3, botInv[2], false));
}

function updateUI(payload){
  state = payload.state;
  legal = payload.legal || [];
  renderBoard(state.board);
  updatePieceDocks();
  const res = payload.result;
  if (res && res!=='ongoing'){
    setStatus(res==='draw' ? 'Draw!' : `${res} wins!`);
    confettiShot();
  } else {
    setStatus(state.to_move===humanSide ? 'Your move' : 'Bot thinking…');
  }
}

async function startNew(){
  const payload = await api('/api/new_game', {human_side: humanSide});
  selectedV = null;
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
  if (selectedV===null){ setStatus('Pick a piece below.'); return; }
  humanMove(selectedV, i);
}

sideX.addEventListener('click', ()=>{ sideX.classList.add('active'); sideO.classList.remove('active'); humanSide='X'; });
sideO.addEventListener('click', ()=>{ sideO.classList.add('active'); sideX.classList.remove('active'); humanSide='O'; });
newGameBtn.addEventListener('click', startNew);

// Confetti with auto-hide
function confettiShot(){
  const cvs = document.getElementById('confetti');
  const ctx = cvs.getContext('2d');
  cvs.style.opacity = 1;
  cvs.width = innerWidth; cvs.height = innerHeight;
  const N = 140;
  const parts = [];
  for (let k=0;k<N;k++){
    parts.push({
      x: Math.random()*cvs.width, y: -10, vy: 2+Math.random()*3,
      vx: (Math.random()-0.5)*2, r: 2+Math.random()*3, a: Math.random()*Math.PI,
      col: Math.random()<0.5 ? '#7c9cff' : '#ff9966'
    });
  }
  let frames = 0;
  (function step(){
    frames++;
    ctx.clearRect(0,0,cvs.width,cvs.height);
    parts.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy; p.a+=0.12;
      ctx.save(); ctx.translate(p.x,p.y); ctx.rotate(p.a);
      ctx.fillStyle = p.col; ctx.fillRect(-p.r,-p.r,2*p.r,2*p.r); ctx.restore();
    });
    if (frames < 120) requestAnimationFrame(step);
    else { cvs.style.opacity = 0; setTimeout(()=>{ ctx.clearRect(0,0,cvs.width,cvs.height); }, 260); }
  })();
}

startNew();
