class ksModel(object):
    """docstring for ClassName"""
    def __init__(self):
        self.d = 0.
        self.N = 0.
        self.h = 0.
        self.nsteps = 0.
        
        
def ksGenerateData(init_cond, ModelParams):
    #Precomputation
    N = ModelParams.N
    d = ModelParams.d
    h = ModelParams.h
    u=0.5*(-1+2*pl.random(N))
    u=u-pl.average(u)
    v=fft(u)

    # Precomputation bit
    k  = pl.concatenate((pl.arange(0, N//2), pl.array([0]), pl.arange(-N//2+1, 0))).T * (2.*pl.pi / d)
    L  = k**2. - k**4.
    E  = pl.exp(h*L)
    E_2= pl.exp(h*L/2.)
    M  = 16
    r  = pl.exp(1j*pl.pi*(pl.arange(1, M+1) - 0.5) *1./M)   # M roots of unity
    LR = h* pl.repeat([L], M, axis=0).T + pl.repeat([r], N, axis=0)
    Q  = h*pl.real(pl.average((pl.exp(LR/2.)-1.)/LR, axis=1))
    f1 = h*pl.real(pl.average((-4.-LR+pl.exp(LR)*(4.-3.*LR+LR**2.))/LR**3., axis=1))
    f2 = h*pl.real(pl.average((2.+LR+pl.exp(LR)*(-2.+LR))/LR**3., axis=1))
    f3 = h*pl.real(pl.average((-4.-3.*LR-LR**2.+pl.exp(LR)*(4.-LR))/LR**3., axis=1))
    g = -0.5j*k
    
    
    # Initial Transient
    u=pl.real(ifft(v))
    for n in range(1,1000):
        Nv = g*fft(pl.real(ifft(v))**2.)
        a = E_2*v + Q*Nv
        Na = g*fft(pl.real(ifft(a))**2.)
        b = E_2*v + Q*Na
        Nb = g*fft(pl.real(ifft(b))**2)
        c = E_2*a + Q*(2.*Nb-Nv)
        Nc = g*fft(pl.real(ifft(c))**2.)
        v = E*v + Nv*f1 + 2.*(Na+Nb)*f2 + Nc*f3
        u=pl.real(ifft(v))
        v=fft(u)

    uinit=pl.copy(u)
    vinit=pl.copy(v)

    # main loop with noise
    data_length = ModelParams.nsteps
    u=pl.copy(uinit)
    v=pl.copy(vinit)
    useries=pl.zeros((N,data_length))
    useries[:,0]=pl.real(ifft(v))
    #~ usum=sum(useries[:,0])
    for n in range(1,data_length):
        Nv = g*fft(pl.real(ifft(v))**2.)
        a = E_2*v + Q*Nv
        Na = g*fft(pl.real(ifft(a))**2.)
        b = E_2*v + Q*Na
        Nb = g*fft(pl.real(ifft(b))**2)
        c = E_2*a + Q*(2.*Nb-Nv)
        Nc = g*fft(pl.real(ifft(c))**2.)
        v = E*v + Nv*f1 + 2.*(Na+Nb)*f2 + Nc*f3
        useries[:,n]=pl.real(ifft(v)) 
        v=fft(useries[:,n])

    xs=pl.copy(useries.T)
