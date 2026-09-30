import json,os,signal,time
pid=os.fork()
if pid:
 print(json.dumps({'descendant_pid':pid}),flush=True)
 os._exit(0)
signal.signal(signal.SIGTERM,signal.SIG_IGN)
time.sleep(60)
