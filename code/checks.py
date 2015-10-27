import cStringIO
import numpy as np
import subprocess
import os

def check_function(f, input, output):
    buf = cStringIO.StringIO()

    fname = f.__name__

    try: 
      for i in input.split("\n"):
        if len(i.rstrip())!=0:
            print >> buf, f(**eval(i))
    except Exception as e:
      print fname, "** DEFINICION INCORRECTA **", e
      return False

    r = buf.getvalue()==output
    if r:
       print fname,"correcto!!"
    else:
       print fname,"** INCORRECTO ... VERIFICA TU CODIGO **"
    return r


def check_script(script, input, output):

    s=cStringIO.StringIO(input)

    p = subprocess.Popen(['python', script], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    poutput, poutput_err = p.communicate(s.read())
    fname = "script '"+script+"'"
    r = output==poutput
    if r:
       print fname,"correcto!!"
    else:
       print fname,"** INCORRECTO ... VERIFICA TU CODIGO **"
    return r
     

