'use strict';
// Exercise the actual client's async submission code with controlled transport races.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(__dirname+'/static/app.js','utf8').split("$('setup').onsubmit=")[0];
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return{promise,resolve,reject};};
function client(){const context=vm.createContext({document:{getElementById:()=>({})},performance,setTimeout});vm.runInContext(source,context);return context;}
(async()=>{
  {
    const context=client(),poll=deferred();
    context.request=async(path)=>path==='/api/new'?{id:'old'}:poll.promise;
    const operation=context.submit('/api/new',{});
    await new Promise(setImmediate);
    vm.runInContext('generation++',context);
    poll.resolve({phase:'done',state:{moves:['e2e4']}});
    assert.equal(await operation,null,'cancelled poll must not commit a completed old board');
  }
  {
    const context=client(),post=deferred();context.request=()=>post.promise;
    const operation=context.submit('/api/new',{});
    vm.runInContext('generation++',context);
    post.reject(new Error('late transport failure'));
    assert.equal(await operation,null,'late POST error must not interfere with a newer operation');
  }
  {
    const context=client(),poll=deferred();let serial=0;
    context.request=async(path)=>path==='/api/new'?{id:++serial===1?'old':'new'}:path.endsWith('old')?poll.promise:{phase:'done',state:{moves:['d2d4']}};
    const old=context.submit('/api/new',{});await new Promise(setImmediate);
    const next=context.submit('/api/new',{});
    assert.equal((await next).moves[0],'d2d4');
    poll.reject(new Error('old poll failed'));
    assert.equal(await old,null,'old poll failure must not unlock or replace the newer board');
  }
  {
    const context=client();let reaction;
    context.mood=(name,line)=>{reaction={name,line};};
    const position=(pieces,human='white',check=false,outcome=null)=>({pieces,human,check,outcome});
    // The human captures, then Chen plays quietly: no celebratory pounce.
    context.react(position(['K','P','k','p']),position(['K','P','k']));
    assert.equal(reaction.name,'idle');
    // Chen captures a human piece, with either human color (en passant included).
    context.react(position(['K','P','k','p']),position(['K','k','p']));
    assert.equal(reaction.name,'pounce');
    context.react(position(['K','P','k','p'],'black'),position(['K','P','k'],'black'));
    assert.equal(reaction.name,'pounce');
    context.react(position(['K','P','k','p'],'black'),position(['K','k','p'],'black'));
    assert.equal(reaction.name,'idle');
    // Quiet move/promotion/castling preserve the piece counts; check and outcome win priority.
    context.react(position(['K','P','k']),position(['K','Q','k']));
    assert.equal(reaction.name,'idle');
    context.react(position(['K','P','k']),position(['K','k'], 'white',true));
    assert.equal(reaction.name,'check');
    context.react(position(['K','P','k']),position(['K','k'], 'white',false,'Chen wins'));
    assert.equal(reaction.name,'win');
  }
  console.log('Client cancellation races and capture reactions: passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
