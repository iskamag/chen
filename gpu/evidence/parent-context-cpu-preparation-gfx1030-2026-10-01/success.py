import os
os.write(1,b"stdout\npartial")
os.write(2,b"stderr\n")
