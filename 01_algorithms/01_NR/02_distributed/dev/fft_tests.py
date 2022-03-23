# %%
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.style.use('default')  # <-- for Jupyter: white figures background


# Define random generator
seed = 12345
rng = np.random.default_rng(seed)

n = 20
z = rng.random(size=(n,))# + 1j * np.random.random(size=(n,))

Z = np.fft.fft(z, n, axis=0)

# Z2 = Z[:int(n/2)+1]
# Z2 = np.concatenate((Z2, [Z[int(n/2)]], np.flip(Z2.conj())[:-1]))

# Z2 = Z[1:int(n/2)+1]
# Z2 = np.concatenate(([Z[0]],Z2, np.flip(Z2[:-1].conj())[:-1]))

Z2 = Z[:int(n/2+1)]
Z2 = np.concatenate((Z2, np.flip(Z2[:-1].conj())[:-1]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Z2.real, 'r')
ax.plot(Z2.imag, 'b')
ax.plot(Z.real, 'g--')
# plt.ylim([-10,10])
ax.plot(Z.imag, 'y--')

z2 = np.fft.ifft(Z2, n)

# print(np.amax(z2.imag))

Z3 = rng.random(size=(int(n/2)+1,)) + 1j * rng.random(size=(int(n/2)+1,))
Z3[0] = Z3[0].real
Z3[-1] = Z3[-1].real
# Z3real = np.concatenate((Z3.real, np.flip(Z3[:-1].real)[:-1]))
# Z3imag = np.concatenate((Z3.imag[:-1], np.flip(Z3.imag)[:]))
# Z3imag = Z3.imag

Z3 = np.concatenate((Z3, np.flip(Z3[:-1].conj())[:-1]))
# Z3p = Z3real + 1j * Z3imag

z3 = np.fft.ifft(Z3, n)
print(np.amax(z3.imag))
fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(Z3.real, 'r')
plt.plot(Z3.imag, 'g')
ax = fig.add_subplot(212)
plt.plot(z3.imag, 'k')

#%% FILTERING IN THE FREQUENCY DOMAIN VS. FILTERING IN THE TIME DOMAIN
import soundfile as sf
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('default')  # <-- for Jupyter: white figures background

RIRfilename = 'myRIR.wav'
soundFilename = 'test_sound.wav'

# Load and pre-process files
x, fs = sf.read(soundFilename)
if x.ndim == 2:
    x = x[:,0]  # make mono
h, fsRIR = sf.read(RIRfilename)
if h.ndim == 2:
    h = h[:,0]  # make mono
if fsRIR != fs:
    # Resample sound
    x = sig.resample(x, int(len(x) * fsRIR / fs))

# Time-domain filtering (convolution)
xh_td = sig.convolve(x, h)

# FFT parameters
n = len(x)   # FFT size [bins]
# Transfer function
H = np.fft.fft(h, n)
# Compute sound spectrum
X = np.fft.fft(x, n)
# Convolution in freq. domain == multiplication
XH_fd = X * H
# Back to time-domain
xh_fd = np.fft.ifft(XH_fd, n)
xh_fd = np.real_if_close(xh_fd)

# Plot
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(411)
ax.plot(x)
ax.grid()
ax.set_title('Dry signal $x$')
ax = fig.add_subplot(412)
ax.plot(h)
ax.grid()
ax.set_title('RIR')
ax = fig.add_subplot(413)
ax.plot(xh_td)
ax.grid()
ax.set_title('Time-domain filtered version of $x$')
ax = fig.add_subplot(414)
ax.plot(xh_fd)
ax.grid()
ax.set_title('Frequency-domain filtered version of $x$')
plt.tight_layout()	
plt.show()

# Print
# print(f'Difference btw. time- and freq.- domain filtered versions of x: {np.mean(np.abs(xh_fd - xh_td[:len(xh_fd)])/np.amax(np.abs(x)))}')
