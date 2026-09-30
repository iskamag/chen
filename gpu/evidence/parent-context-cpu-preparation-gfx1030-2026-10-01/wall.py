import signal,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
time.sleep(60)
