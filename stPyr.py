import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

###############################################################
# Polar grid with Nyquist corrections
###############################################################
""" polarGridCorrected
    This function generates a polar grid of the Fourier spectr.
    it contains some patches to enforce symmetry, which is
    important in the case of even image sizes, as this hapers
    symmetry, which introduces reconstruction errors.
"""
def polarGridCorrected(H, W, device):
    # Standard FFT freqs
    fy = torch.fft.fftfreq( H, device=device )
    fx = torch.fft.fftfreq( W, device=device )

    Fy = fy.view( H, 1 ).expand(H, W)
    Fx = fx.view( 1, W ).expand(H, W)

    # Raw polar values
    RHO = torch.sqrt( Fx**2 + Fy**2 )
    PHI = torch.atan2( Fy, Fx )

    # Identify problematic Nyquist bins
    nyq_mask = (
        ( Fx.abs() == 0.5 ) |
        ( Fy.abs() == 0.5 )
    )

    # Fix angles at Nyquist bins (value irrelevant as long as symmetric)
    PHI[nyq_mask] = 0.0

    # Fix magnitudes at Nyquist bins (copy nearest valid)
    eps = 1e-9
    RHO[nyq_mask] = ( 0.5 - eps )
    return( RHO, PHI )


###############################################################
# Simoncelli Radial Filters
###############################################################
""" radialPartition
    This class implements the radial partition of Simoncelli's
    steerable pyramid. For the recursive implementation, this
    partition is only implemented once.
"""
class radialPartition(nn.Module):
    def __init__(self, whichScale=1, cut_freq = 0.25, twidth=1.0, a=2):
        super().__init__()
        # Setting transition width
        self.a = a
        self.twidth = twidth
        self.whichScale = whichScale
        # Initial cut frequency in cycles/pixel
        self.cut_freq = cut_freq
        # For device
        self._dummy = nn.Parameter(torch.empty(0))

    """ forward
    Inputs
        * rho: Angular mask
    Outputs
        * Low and high pass bands
    """
    def forward(self, rho):
        # We dilate based on the scale. This is exclusively for undecimated transforms
        a = self.a
        scale = a**(self.whichScale)
        rho_s = torch.clamp(rho * scale, 0, 0.5 - 1e-9)
        
        low, hig = self.simoncelli_medium(rho_s, cut_freq = self.cut_freq, twidth= self.twidth)
        #low, hig  = self.waveletSimoncelli_old(rho=rho, radial_val = 1/scale )
        return(hig, low)

    # The default value sets cut frequency at 0.25 cycles/pixel
    # Isotropic primal wavelet
    def simoncelli_medium(self, rho, cut_freq = 2**(-2), twidth = 1.0 ):
        log_rad = torch.log2(rho) - np.log2(cut_freq)
        hi_mask = torch.clamp(log_rad, -twidth, 0)
        #hi_mask = torch.clamp(torch.cos( hi_mask*np.pi/(2*twidth)) ,0,1 )
        hi_mask = torch.cos( hi_mask*np.pi/(2*twidth))
        lo_mask = torch.sqrt(1.0 - hi_mask**2)
        return(lo_mask, hi_mask)


    def test(self, dSize = 32):
        # Setting device
        dev = self.device()
        ds = dSize

        # Defininig a polar grid
        rho, _ = polarGridCorrected( ds, ds, device=dev)

        # Applying radial partition
        low, hig = self.simoncelli_medium( rho, twidth=1 )
        su = low**2 + hig**2
        err = torch.max((1 - su).abs() )
        if err > 2e-7:
            raise( RuntimeError("Reconstruction error is too high") )
        else:
            print("Test passed, error: %s"%(err) )
            return(True)
        
    """ Device
    Outputs: Device in which the module is executed
    """
    def device(self):
        return( self._dummy.device )



###############################################################
# Corrected Angular Partition (Even order + symmetry)
###############################################################
""" AngularPartition
    This class implements the angular partition of Simoncelli's
    steerable pyramid. It produces only real filters. In order 
    to enforce real filters it usesalways an even order. 
"""
class AngularPartition(nn.Module):
    """ __init__
    Inputs
        * orientations: Number of orientations
    """
    def __init__(self, orientations):
        super().__init__()
        self.orientations = orientations
        self.order = orientations - 1

        # Force even order for real steerable pyramid
        if self.order % 2 == 1:
            self.order -= 1
        # Dummy variable to determine device
        self._dummy = nn.Parameter(torch.empty(0))

    """ forward
    Inputs
        * forward: Angular mask
    Outputs
        * Angular band pass masks
    """
    def forward(self, phi):
        K = self.orientations
        N = self.order

        angle_masks = []
        for k in range(K):
            theta_k = np.pi * k / K
            dphi = torch.remainder(phi - theta_k + np.pi, 2*np.pi) - np.pi
            
            # Raised cosine
            Ak = np.abs(torch.cos(dphi) ** N)
            
            angle_masks.append(Ak.unsqueeze(0))

        angs = torch.cat(angle_masks, dim=0)

        # Sum of squares normalization
        # Easier and works better than analytic solution
        norm = torch.sqrt((angs**2).sum(dim=0, keepdim=True) + 1e-12)
        angs = angs / norm

        return( angs )
    
    """ Device
    Outputs: Device in which the module is executed
    """
    def device(self):
        return( self._dummy.device )


###############################################################
# Steerable Filter for one level
###############################################################
""" SteerableFilter1lvl
It implements one decomposition of the steerable filter transf.
This module is used recursively to generate multiple-scales. 

Inputs:
    * Y, Y_low: Decomposed band-pass and low-pass residual.
        The values are on the spatial domain
Output:
    * Reconstructed signal
""" 
class SteerableFilter1lvl(nn.Module):
    """ __init__
    Inputs:
        * noOrientations: Number of orientations of the steerable filter
        * whichScale: Scale being implemented
        * cut_freq: Cut frequency in cycles/pixel for the low-pass filter at the coarsest scale
        * twidth: Size of the transition width. The closest to zero, the more "squared" the frequency
            partition, which causes filters with more cycles in the spatial domain. The larger the 
            number, the faster the filter decays in the spatial domain
    Output:
        * Reconstructed signal
    """ 
    def __init__(self, noOrientations, whichScale=0, cut_freq=0.25, twidth=1.0, a=2):
        super().__init__()
        # Setting up the parameters
        self.noOrientations = noOrientations       

        # Calling the radial and angular partitions
        self.scales = radialPartition(whichScale, cut_freq, twidth, a)
        self.angles = AngularPartition(noOrientations)

        # Dummy variable to determine device
        self._dummy = nn.Parameter(torch.empty(0))


    """ filter_masks: This function generates the actual masks based on the frequency
    and angular partitions
    Inputs:
        * N1, N2: Dimensions of the mask
        * device: Device in which the model is running.
    Output:
        * filters: High-pass residual and bandpass filters
        * lowpass: lowpass residual
    """ 
    def filter_masks(self, N1, N2, device):
        # Generating polar grid
        RHO, PHI = polarGridCorrected(N1, N2, device)

        # Generating radual and angular envelopes
        hhigh, hlow = self.scales(RHO)
        angs = self.angles(PHI)

        # Applying masks and reshaping
        filters = hhigh.unsqueeze(0) * angs
        filters = filters.reshape(1, -1, N1, N2)
        lowpass = hlow.unsqueeze(0)
        return(filters, lowpass)

    # Forward applies analysis function
    def forward(self, x):
        return self.apply(x)

    """ transpose
    It implements synthesis/adjoint
    Inputs:
        * Y, Y_low: Decomposed band-pass and low-pass residual.
            The values are on the spatial domain
    Output:
        * Reconstructed signal
    """ 
    # Analysis
    def apply(self, x):
        # Generating masks
        filters, lowpass = self.filter_masks(
            x.size(-2), x.size(-1), x.device)
        self.filters, self.lowpass = (filters, lowpass)

        # FFT of inpput
        X = torch.fft.fft2(x)
        
        # Applying filters and inverse transforming
        Y = torch.fft.ifft2(X * filters).real 
        Y_low = torch.fft.ifft2(X * lowpass).real
        return(Y, Y_low)
        

    """ transpose
    It implements synthesis/adjoint
    Inputs:
        * Y, Y_low: Decomposed band-pass and low-pass residual.
            The values are on the spatial domain
    Output:
        * Reconstructed signal
    """ 
    def transpose(self, Y, Y_low):
        # Computing filter masks
        filters, lowpass = (self.filters, self.lowpass)

        # FFT of inputs
        Yf = torch.fft.fft2(Y)
        Ylf = torch.fft.fft2(Y_low)

        # Applying masks and adding everything
        Xf = (filters * Yf).sum(dim=1, keepdim=True) + (lowpass * Ylf)

        return(torch.fft.ifft2(Xf).real)

    """ test
    Implements unit testing by comparing the impulse response of the
    entire pyramid and is input impulse.
    """ 
    def test(self, ds=32):
        # Delta impulse
        dirac = torch.zeros((1,1,ds,ds), device=self.device())
        dirac[:,:,ds//2, ds//2] = 1

        # Forward and inverse decomposition
        bp, lp = self.forward(dirac)
        rec = self.transpose(bp, lp)

        # Computing absoule error
        err = (dirac - rec).abs().max()
        if err > 2e-7:
            raise RuntimeError("Reconstruction error is too high")
        else:
            print("Test passed, reconstruction error:", err.item())
            return(True)

    def device(self):
        return self._dummy.device

###############################################################
# Multi-level steerable filter
###############################################################
""" SteerablePyramid
It implements synthesis/adjoint of a multi-scale steerable filter. It allows to implemen
both, decimated and undecimated steerable filters
Inputs:
    * noScales: Number of scales of the steerable pyramid decomposition
    * noOrientations: Number of angular filters per scale
    * cut_freq: Cut frequency of the high/low-pass filters at every scale. If multi-resolution
        is needed, then this has to be set to 0.25. Otherwise, aliasing will be introduced in 
        the signal.
    * twidth: Transition width for the high/low-pass filters
"""
class SteerablePyramid(nn.Module):
    def __init__(self, noScales, noOrientations,  cut_freq=0.25, twidth=1.0, decimate=False,a=2):
        super().__init__()
        # Setting up the parameters
        self.stf = nn.ModuleList()
        for s in range(noScales):
            if decimate:
                self.stf.append(
                    SteerableFilter1lvl(
                        noOrientations, whichScale=0, cut_freq=0.25, twidth=twidth, a=2))
            else:
                self.stf.append(
                    SteerableFilter1lvl(
                        noOrientations, whichScale=s, cut_freq=cut_freq, twidth=twidth, a=a))
        self.decimate=decimate

    def forward(self, x):
        bps = []
        x_prev = x
        for st in self.stf:
            bp, lo = st(x_prev)
            bps.append(bp)
            if self.decimate:
                x_prev =  self.down2(lo)
            else:
                x_prev =  lo
        return(bps, x_prev)

    def transpose(self, x, x_low):
        x_prev = x_low
        for i, st in enumerate(reversed(self.stf), start=1):
            indx = len(self.stf)-i
            if self.decimate:
                x_prev = st.transpose(x[indx], 4*self.up2(x_prev))
            else:
                x_prev = st.transpose(x[indx], x_prev)
        return(x_prev)    

    def down2(self, x):
        """
        Downsample x by factor 2 in the spatial domain.
        x: tensor of shape (B, C, H, W)
        """
        return(x[:, :, ::2, ::2])

    def up2(self, x):
        """
        Upsample x by factor 2 using zero insertion.
        x: tensor of shape (B, C, H, W)
        Returns shape (B, C, 2H, 2W)
        """
        B, C, H, W = x.shape
        y = torch.zeros(B, C, H*2, W*2, device=x.device, dtype=x.dtype)
        y[:, :, ::2, ::2] = x
        return(y)



    import torch.nn as nn



def frameSq(x):
    X = torch.fft.fft2(x)
    Y = (X.conj()*X)
    return(torch.fft.ifft2(Y).real)


class stp_fast_recon(nn.Module):
    def __init__(self, noSc, noOri,  cut_freq=0.25, a=2, ks=31, fastRecon=True):
        super().__init__()
        self.fast = fastRecon
        stp = SteerablePyramid(
            noScales=noSc,
            noOrientations=noOri,
            cut_freq=cut_freq,
            decimate=False,
            a=a)# size of equivalent kernel
        
        # Building a dirac's delta
        dirac = torch.zeros((1,1,ks,ks), requires_grad=False)
        dirac[0,0,0,0] = 1
        dirac = torch.fft.fftshift(dirac)

        # Computing impulse response
        with torch.no_grad():
            Wx, WxLow = stp(dirac)
            W = torch.cat(Wx, dim=1)
            W2 = torch.fft.fftshift(frameSq(W ), dim=(-2,-1) ) 
            W2l = torch.fft.fftshift(frameSq( WxLow), dim=(-2,-1) )

        # For conventional reconstruction

        self.register_buffer(
            "W",
            torch.as_tensor(W, dtype=torch.float32))
        self.register_buffer(
            "Wl",
            torch.as_tensor(WxLow, dtype=torch.float32))
        self.register_buffer(
            "W2",
            torch.as_tensor(W2, dtype=torch.float32))
        self.register_buffer(
            "W2l",
            torch.as_tensor(W2l, dtype=torch.float32))

    def forward(self, x):
        if self.fast:
            w = self.W2.shape[-1]

            # Reshaping, such that the same transform is applied to all channels
            p = w//2
            xs = x.shape
            x = x.reshape(xs[0]*xs[1], 1, xs[2], xs[3])
            x_pad = torch.nn.functional.pad(
                x, (p, p, p, p), mode="circular")

            wx = F.conv2d(x_pad, self.W2.transpose(1,0))
            wxl = F.conv2d(x_pad, self.W2l)

            # Reshaping to correct dimensions and retrieving
            wx = wx.reshape(xs[0], xs[1]*self.W2.shape[1], xs[2], xs[3])
            wxl = wxl.reshape(*xs)
            return([ wx, wxl])
        
        else:
            # Reshaping, such that the same transform is applied to all channels
            xs = x.shape
            x = x.reshape(xs[0]*xs[1], 1, xs[2], xs[3])

            w = self.W.shape[-1]
            p = w//2
            x_pad = torch.nn.functional.pad(
                x, (p, p, p, p), mode="circular")

            wx = F.conv2d(x_pad, self.W.transpose(1,0))
            wxl = F.conv2d(x_pad, self.Wl)

            # Reshaping to correct dimensions and retrieving
            wx = wx.reshape(xs[0], xs[1]*self.W2.shape[1], xs[2], xs[3])
            wxl = wxl.reshape(*xs)
            return([wx, wxl])
        

    def recon(self, wx, wxl):
        if self.fast:
            B = wx.shape[0]
            nob = self.W2.shape[1]
            C = wx.shape[1] // nob
            H = wx.shape[2]
            W = wx.shape[3]

            wx = wx.view(B, C, nob, H, W)
            xr = wx.sum(dim=2)

            return xr + wxl
        else:
            B = wx.shape[0]
            nob = self.W2.shape[1]
            C = wx.shape[1] // nob
            H = wx.shape[2]
            W = wx.shape[3]
            wx = wx.view(B*C, nob, H, W)
            wxl = wxl.view(B*C, 1, H, W)

            p = self.W.shape[-1]//2
            wx_pad = torch.nn.functional.pad(
                wx, (p, p, p, p), mode="circular")
            wlx_pad = torch.nn.functional.pad(
                wxl, (p, p, p, p), mode="circular")

            wtwx = F.conv2d(wx_pad, torch.flip(self.W, dims=(-2, -1)) ,stride=1, dilation=1)
            wltwxl = F.conv2d(wlx_pad, torch.flip(self.Wl, dims=(-2, -1)), stride=1, dilation=1)
            rec = wltwxl+wtwx
            return rec.reshape(B,C,H,W)