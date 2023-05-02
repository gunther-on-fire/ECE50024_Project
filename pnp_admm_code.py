import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2
import bm3d

def proj(x):
# % x     = input vector
# % bound = 2x1 vector
# %
# % Example: out = proj_bound(x, [1,3]);
# % projects a vector x onto the interval [1,3]
# % by setting x(x>3) = 3, and x(x<1) = 1
# %
# % 2016-07-24 Stanley Chan

    bound = [0,1]
    over_ceiling_idx = x > bound[1]
    x[over_ceiling_idx] = 1
    under_floor_idx = x < bound[0]
    x[under_floor_idx] = 0
    return x


def fspecial_gauss(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def psnr(img1, img2):
    return -10 * np.log10(np.mean((img1-img2) ** 2))


def calc_rmse(img1, img2):
    return np.sqrt(np.square(np.subtract(img1, img2)).mean())


def corrupt_image(img, ker, noise_lvl):
    y = scipy.ndimage.correlate(img, ker, mode='wrap')
    y = y + noise_lvl * np.random.randn(len(y), len(y[0]))
    y = proj(y)
    return y    


# Methods
def ridge_regression(img_corr, ker, lambd):
    dim = img_corr.shape 

    Hty = scipy.ndimage.correlate(img_corr, ker, mode='wrap')
    eigHtH = np.abs(np.fft.fftn(ker, dim)) ** 2
    rhs = np.fft.fftn(Hty, dim)
    return np.real(np.fft.ifftn(rhs/(eigHtH + lambd), dim))


def pnp_admm_deblur(y, h, lambd, opts):

    # check defaults
    if 'rho' not in opts:
        opts['rho'] = 1
    if 'max_itr' not in opts:
        opts['max_itr'] = 20
    if 'tol' not in opts:
        opts['tol'] = 1e-3
    if 'gamma' not in opts:
        opts['gamma'] = 1
    if 'print' not in opts:
        opts['print'] = False
    if 'eta' not in opts:
        opts['eta'] = None
    
    # set parameters
    max_itr   = opts['max_itr']
    tol       = opts['tol']
    gamma     = opts['gamma']
    rho       = opts['rho']
    eta = opts['eta'] # adaptive rule

    # initialize variables
    dim = y.shape   
    N = dim[0] * dim[1]

    Hty = scipy.ndimage.correlate(y, h, mode='wrap')
    eigHtH = np.abs(np.fft.fftn(h, dim)) ** 2;

    v = 0.5 * np.ones(dim)
    x = v
    u = np.zeros(dim)
    residual = float("inf")

    # main loop
    if opts['print'] == True:
        print('Plug-and-Play ADMM --- Deblurring \n')
        print('itr \t ||x-xold|| \t ||v-vold|| \t ||u-uold|| \n')

    itr = 1
    while (residual > tol and itr <= max_itr):
        # store x, v, u from previous iteration for psnr residual calculation
        x_old = x
        v_old = v
        u_old = u
    
        # inversion step
        xtilde = v - u
        rhs = np.fft.fftn(Hty + rho * xtilde, dim)
        x = np.real(np.fft.ifftn(rhs/(eigHtH + rho), dim))
    
        # denoising step
        vtilde = x + u
        vtilde = proj(vtilde)
        sigma  = np.sqrt(lambd/rho)
        v = bm3d.bm3d(vtilde, sigma)
    
        # update langrangian multiplier
        u = u + (x - v)
    
        # calculate residual
        residualx = (1/np.sqrt(N))*(np.sqrt(np.sum(np.sum((x - x_old)**2))))
        residualv = (1/np.sqrt(N))*(np.sqrt(np.sum(np.sum((v - v_old)**2))))
        residualu = (1/np.sqrt(N))*(np.sqrt(np.sum(np.sum((u - u_old)**2))))
    
        residual_upd = residualx + residualv + residualu

        if eta == None:
            rho = rho * gamma
        elif residual_upd >= eta * residual:
            rho = rho * gamma

        residual = residual_upd 
         
        if opts['print'] == True:
            print(itr, residualx, residualv, residualu)
    
        itr += 1
    
    return v

# Perform the experiments

# read test image
test_img = cv2.imread('./data/Cameraman256.png', cv2.IMREAD_GRAYSCALE)
test_img_norm = cv2.normalize(test_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# (1) Use ridge regression to find an optimal value for the regularization parameter

# (2) Fix PSF and vary noise levels

# Generate the list of regularization parameters
lambd = np.arange(0.0001, 1, step=0.001)

# Specify the kernel
h = fspecial_gauss(9, 1)

# Generate noise levels
MAX_PIX = 255
noise_lvls = [5/MAX_PIX, 10/MAX_PIX, 15/MAX_PIX, 20/MAX_PIX, 25/MAX_PIX]

# (2a) Estimate optimal lambdas for different noise levels
plt.figure(1)
plt.ylim(0, 0.6)
plt.xlabel(r'$\lambda$')
plt.ylabel('RMSE')

lambd_opt = [] # list to contain optimal lambdas

noise_idx = 1
for noise in noise_lvls:
    y_corrupted = corrupt_image(test_img_norm, h, noise)

    rmse_lambd_rr = [] # list to store RMSE for optimal lambdas
    
    for l in lambd:
        x_rr = ridge_regression(y_corrupted, h, l)
        rmse_rr = calc_rmse(x_rr, test_img_norm)
        rmse_lambd_rr.append(rmse_rr)

    lambd_opt.append(lambd[np.argmin(rmse_lambd_rr)])

    print(f'noise level: {noise:.4f}, result: {lambd[np.argmin(rmse_lambd_rr)]}')

    plt.plot(lambd, rmse_lambd_rr, label=f'noise level {noise_idx}')
    plt.plot(lambd[np.argmin(rmse_lambd_rr)], rmse_lambd_rr[np.argmin(rmse_lambd_rr)], 'o', 
             label=f'$\lambda_{{opt {noise_idx}}}$ = {lambd[np.argmin(rmse_lambd_rr)]:.4f}')
    noise_idx += 1

plt.legend(loc=1)
plt.savefig('rmse_vs_lambda.png')

# The list of optimal parameters for different noise levels
print(lambd_opt)


# "Tune" PnP ADMM for each "optimal" lambda
# (2b) Looking for a "good" value of gamma
# It turns out that gamma should be from 2 to 4

opts = {}
opts['rho']     = 1e-3 # fixed rho (according to the paper)
opts['max_itr'] = 20 # according to the paper
opts['print']   = True
opts['eta'] = None # without the adaptive rule

# Experiment with gamma
gamma_list = [1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 15]

plt.figure(2)
plt.xlabel(r'$\lambda$')
plt.ylabel('RMSE')

for gamma in gamma_list:
    opts['gamma'] = gamma

    rmse_pnp_list = []
    for i in range(len(noise_lvls)):
        y_corrupted = corrupt_image(test_img_norm, h, noise_lvls[i])

        x_pnp = pnp_admm_deblur(y_corrupted, h, lambd_opt[i], opts)

        rmse_pnp = calc_rmse(x_pnp, test_img_norm)
        rmse_pnp_list.append(rmse_pnp)

    print(f'RMSE for PnP: {rmse_pnp_list}')
    plt.plot(lambd_opt, rmse_pnp_list, marker='.', label=f'$\gamma$={gamma}')

plt.legend(loc=1)
plt.savefig('rmse_vs_lambda_gamma_opt.png')

# (2c) Now gamma is fixed, looking for the "best"
# eta in the continuation scheme with the adaptive rule

plt.figure(3) 
plt.xlabel(r'$\lambda$')
plt.ylabel('RMSE')

opts = {}
opts['rho']     = 1e-3 # fixed rho (according to the paper)
opts['max_itr'] = 20 # according to the paper
opts['print']   = True
opts['gamma'] = 2.5 # fix "optimal" gamma
eta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for eta in eta_list:
    opts['eta'] = eta

    rmse_pnp_list = []
    for i in range(len(noise_lvls)):
        y_corrupted = corrupt_image(test_img_norm, h, noise_lvls[i])

        x_pnp = pnp_admm_deblur(y_corrupted, h, lambd_opt[i], opts)

        rmse_pnp = calc_rmse(x_pnp, test_img_norm)
        rmse_pnp_list.append(rmse_pnp)

    print(f'RMSE for PnP: {rmse_pnp_list}')
    plt.plot(lambd_opt, rmse_pnp_list, marker='.', label=f'$\eta$={eta}')

plt.legend(loc=1)
plt.savefig('rmse_vs_lambda_eta_opt.png')

# (3) Compare image restoration methods
psnr_rr = []
rmse_rr = []
for i in range(len(noise_lvls)):
    y_corrupted = corrupt_image(test_img_norm, h, noise_lvls[i])

    x_rr = ridge_regression(y_corrupted, h, lambd_opt[i])

    psnr_out = psnr(x_rr, test_img_norm)
    psnr_rr.append(psnr_out)
    psnr_val = 'PSNR = ' + str(np.round(psnr_out, 3)) + ' dB'

    rmse_val = calc_rmse(x_rr, test_img_norm)
    rmse_rr.append(rmse_val)

    plt.figure()
    plt.subplot(121)
    plt.title(f'Degraded Image, noise level {i+1}')
    plt.imshow(y_corrupted, cmap='gray')

    plt.subplot(122)
    plt.imshow(x_rr, cmap='gray')
    plt.title('Restored image, ' + psnr_val)
    plt.tight_layout()
    
    plt.savefig('rec_rr_' + str(i) + '.png' )


# Based on experiments
opts = {}
opts['rho']     = 1e-3 # fixed rho (according to the paper)
opts['max_itr'] = 20 # according to the paper
opts['print']   = True
opts['eta'] = None # without the adaptive rule
opts['gamma'] = 2.5

psnr_pnp_mono = []
rmse_pnp_mono = []

for i in range(len(noise_lvls)):
    y_corrupted = corrupt_image(test_img_norm, h, noise_lvls[i])

    x_pnp = pnp_admm_deblur(y_corrupted, h, lambd_opt[i], opts)

    psnr_out = psnr(x_pnp, test_img_norm)
    psnr_pnp_mono.append(psnr_out)
    psnr_val = 'PSNR = ' + str(np.round(psnr_out, 3)) + ' dB'

    rmse_val = calc_rmse(x_pnp, test_img_norm)
    rmse_pnp_mono.append(rmse_val)

    plt.figure()
    plt.subplot(121)
    plt.title(f'Degraded Image, noise level {i+1}')
    plt.imshow(y_corrupted, cmap='gray')

    plt.subplot(122)
    plt.imshow(x_pnp, cmap='gray')
    plt.title('Restored image, ' + psnr_val)
    plt.tight_layout()

    plt.savefig('rec_pnp_mono_' + str(i) + '.png' )

# Parameters based on experiments
opts = {}
opts['rho']     = 1e-3 # fixed rho (according to the paper)
opts['max_itr'] = 20 # according to the paper
opts['print']   = True
opts['eta'] = 0.5
opts['gamma'] = 2.5

psnr_pnp_adapt = []
rmse_pnp_adapt = []

for i in range(len(noise_lvls)):
    y_corrupted = corrupt_image(test_img_norm, h, noise_lvls[i])

    x_pnp = pnp_admm_deblur(y_corrupted, h, lambd_opt[i], opts)

    psnr_out = psnr(x_pnp, test_img_norm)
    psnr_pnp_adapt.append(psnr_out)
    psnr_val = 'PSNR = ' + str(np.round(psnr_out, 3)) + ' dB'

    rmse_val = calc_rmse(x_pnp, test_img_norm)
    rmse_pnp_adapt.append(rmse_val)

    plt.figure()
    plt.subplot(121)
    plt.title(f'Degraded Image, noise level {i+1}')
    plt.imshow(y_corrupted, cmap='gray')

    plt.subplot(122)
    plt.imshow(x_pnp, cmap='gray')
    plt.title('Restored image, ' + psnr_val)
    plt.tight_layout()

    plt.savefig('rec_pnp_adapt_' + str(i) + '.png' )

plt.figure()
plt.xlabel('Noise Level')
plt.ylabel('RMSE')
plt.plot(noise_lvls, rmse_rr, marker='.', label='Ridge Regression')
plt.plot(noise_lvls, rmse_pnp_mono, marker='o', label='PnP ADMM monotone')
plt.plot(noise_lvls, rmse_pnp_adapt, marker='^', label='PnP ADMM adaptive')
plt.legend(loc=1)
plt.savefig('rmse_res.png')

plt.figure()
plt.xlabel('Noise Level')
plt.ylabel('PSNR')
plt.plot(noise_lvls, psnr_rr, marker='.', label='Ridge Regression')
plt.plot(noise_lvls, psnr_pnp_mono, marker='o', label='PnP ADMM monotone')
plt.plot(noise_lvls, psnr_pnp_adapt, marker='^', label='PnP ADMM adaptive')
plt.legend(loc=1)
plt.savefig('psnr_res.png')

plt.show()