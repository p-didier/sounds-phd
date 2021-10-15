import glob
from PIL import Image

def makegif(mydir, gifname, dur=100, loopme=0):
    # Exports a GIF from the PNG files present in a directory.
    
    imgs = glob.glob('%s\\*.png' % mydir)
    frames = []
    for ii in imgs:
        frames.append(Image.open(ii))
    frames[0].save('%s\\%s.gif' % (mydir, gifname), format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=dur, loop=loopme)

    return None